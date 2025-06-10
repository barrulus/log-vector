#!/usr/bin/env python3
"""
Trust Remote Code Manager

Handles detection of models requiring trust_remote_code and manages user consent.
Saves approval state in .env with model tracking for security.
"""

import os
from typing import Optional, TYPE_CHECKING
from dotenv import load_dotenv

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()


class TrustManager:
    """Manages trust_remote_code consent and detection"""
    
    def __init__(self, env_file: str = '.env'):
        self.env_file = env_file
        self._ensure_env_file()
    
    def _ensure_env_file(self) -> None:
        """Ensure .env file exists"""
        if not os.path.exists(self.env_file):
            with open(self.env_file, 'w') as f:
                f.write("# Trust Remote Code Settings\n")
                f.write("# Format: TRUST_REMOTE_CODE_<MODEL_HASH>=true|false\n\n")
    
    def _model_to_key(self, model_name: str) -> str:
        """Convert model name to environment variable key"""
        # Create a hash of the model name for consistent key generation
        import hashlib
        model_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
        return f"TRUST_REMOTE_CODE_{model_hash.upper()}"
    
    def _get_model_approval(self, model_name: str) -> Optional[bool]:
        """Check if model has been approved for trust_remote_code"""
        key = self._model_to_key(model_name)
        value = os.getenv(key)
        if value is None:
            return None
        return value.lower() == 'true'
    
    def set_model_approval(self, model_name: str, approved: bool) -> None:
        """Save model approval to .env file"""
        key = self._model_to_key(model_name)
        value = 'true' if approved else 'false'
        
        # Add comment with model name for clarity
        comment_key = f"# {key}_MODEL"
        
        # Read existing content
        env_content: list[str] = []
        if os.path.exists(self.env_file):
            with open(self.env_file, 'r') as f:
                env_content = f.readlines()
        
        # Remove existing entries for this model
        filtered_content: list[str] = []
        skip_next = False
        for line in env_content:
            if skip_next:
                skip_next = False
                continue
            if line.strip() == f"{comment_key}={model_name}":
                skip_next = True  # Skip the next line (the actual setting)
                continue
            if not line.startswith(key + '='):
                filtered_content.append(line)
        
        # Add new entries
        filtered_content.append(f"{comment_key}={model_name}\n")
        filtered_content.append(f"{key}={value}\n")
        
        # Write back to file
        with open(self.env_file, 'w') as f:
            f.writelines(filtered_content)
        
        # Reload environment
        load_dotenv(override=True)
    
    def check_model_needs_trust(self, model_name: str) -> bool:
        """
        Auto-detect if a model requires trust_remote_code.
        This is a heuristic based on known model patterns.
        """
        # Known models that require trust_remote_code
        trust_required_patterns = [
            'nomic-ai/',
            'microsoft/codebert',
            'sentence-transformers/all-mpnet-base-v2',
            # Add more patterns as needed
        ]
        
        # Known models that don't require trust_remote_code
        safe_patterns = [
            'sentence-transformers/all-MiniLM',
            'sentence-transformers/paraphrase',
            'sentence-transformers/multi-qa',
        ]
        
        model_lower = model_name.lower()
        
        # Check safe patterns first
        for pattern in safe_patterns:
            if pattern.lower() in model_lower:
                return False
        
        # Check trust required patterns
        for pattern in trust_required_patterns:
            if pattern.lower() in model_lower:
                return True
        
        # For unknown models, assume trust might be needed
        return True
    
    def prompt_user_consent(self, model_name: str) -> bool:
        """Prompt user for consent to use trust_remote_code"""
        print(f"\n{'='*60}")
        print("SECURITY WARNING: Remote Code Execution")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print()
        print("This model may require 'trust_remote_code=True' which allows")
        print("the model to execute arbitrary code during loading.")
        print()
        print("RISKS:")
        print("- The model could execute malicious code")
        print("- Your system could be compromised")
        print("- Data could be stolen or corrupted")
        print()
        print("BENEFITS:")
        print("- Access to newer/specialized models")
        print("- Better embedding quality for some models")
        print()
        print("Your choice will be saved for this model.")
        print(f"{'='*60}")
        
        while True:
            try:
                response = input("Allow remote code execution for this model? [y/N]: ").strip().lower()
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no', '']:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return False
    
    def get_trust_setting(self, model_name: str, interactive: bool = True) -> bool:
        """
        Get trust_remote_code setting for a model.
        
        Args:
            model_name: Name of the model
            interactive: Whether to prompt user if not already approved
            
        Returns:
            True if trust_remote_code should be used, False otherwise
        """
        # Check if we have a saved decision
        approval = self._get_model_approval(model_name)
        if approval is not None:
            return approval
        
        # Check if model needs trust_remote_code
        needs_trust = self.check_model_needs_trust(model_name)
        if not needs_trust:
            # Model is known safe, save this decision
            self.set_model_approval(model_name, False)
            return False
        
        # Model might need trust, prompt user if interactive
        if interactive:
            user_approved = self.prompt_user_consent(model_name)
            self.set_model_approval(model_name, user_approved)
            return user_approved
        else:
            # Non-interactive mode, default to False for security
            print(f"Warning: Model {model_name} may require trust_remote_code but running in non-interactive mode.")
            print("Defaulting to False for security. Use interactive mode to approve.")
            return False
    
    def list_approved_models(self) -> dict[str, bool]:
        """List all models and their approval status"""
        approved_models: dict[str, bool] = {}
        
        # Read .env file and find trust settings
        if os.path.exists(self.env_file):
            with open(self.env_file, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('# TRUST_REMOTE_CODE_') and line.endswith('_MODEL'):
                    # This is a model comment line
                    model_name = line.split('=', 1)[1] if '=' in line else 'Unknown'
                    # Next line should be the setting
                    if i + 1 < len(lines):
                        setting_line = lines[i + 1].strip()
                        if '=' in setting_line:
                            approved = setting_line.split('=', 1)[1].lower() == 'true'
                            approved_models[model_name] = approved
                    i += 2
                else:
                    i += 1
        
        return approved_models


def safe_sentence_transformer_load(model_name: str, interactive: bool = True, **kwargs: object) -> 'SentenceTransformer':
    """
    Safely load a SentenceTransformer with trust_remote_code consent management.
    
    Args:
        model_name: Name of the model to load
        interactive: Whether to prompt user for consent
        **kwargs: Additional arguments to pass to SentenceTransformer
        
    Returns:
        Loaded SentenceTransformer model
    """
    from sentence_transformers import SentenceTransformer
    
    trust_manager = TrustManager()
    trust_setting = trust_manager.get_trust_setting(model_name, interactive)
    
    # Remove any existing trust_remote_code setting from kwargs
    kwargs_dict = dict(kwargs)
    kwargs_dict.pop('trust_remote_code', None)
    
    if trust_setting:
        print(f"Loading {model_name} with trust_remote_code=True (user approved)")
        return SentenceTransformer(model_name, trust_remote_code=True, **kwargs_dict)  # type: ignore[misc]
    else:
        print(f"Loading {model_name} with trust_remote_code=False")
        try:
            return SentenceTransformer(model_name, trust_remote_code=False, **kwargs_dict)  # type: ignore[misc]
        except Exception as e:
            print(f"Error loading model with trust_remote_code=False: {e}")
            if interactive:
                print("Model may require trust_remote_code=True. Would you like to try again with remote code enabled?")
                if trust_manager.prompt_user_consent(model_name):
                    trust_manager.set_model_approval(model_name, True)
                    return SentenceTransformer(model_name, trust_remote_code=True, **kwargs_dict)  # type: ignore[misc]
            raise


if __name__ == "__main__":
    # CLI for managing trust settings
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage trust_remote_code settings")
    parser.add_argument('--list', action='store_true', help='List approved models')
    parser.add_argument('--check', type=str, help='Check if model needs trust_remote_code')
    
    args = parser.parse_args()
    
    trust_manager = TrustManager()
    
    if args.list:
        models = trust_manager.list_approved_models()
        if models:
            print("Approved models:")
            for model, approved in models.items():
                status = "✓ Approved" if approved else "✗ Denied"
                print(f"  {model}: {status}")
        else:
            print("No models have been evaluated yet.")
    
    elif args.check:
        needs_trust = trust_manager.check_model_needs_trust(args.check)
        print(f"Model {args.check} {'likely needs' if needs_trust else 'probably does not need'} trust_remote_code")
    
    else:
        parser.print_help()