import logging
from src.pipeline import CredentialGenerationPipeline
from src.tuning import tune_hyperparameters, TuneableCredentialPipeline
from src.credentials_loader import load_credentials
import argparse
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_and_generate(training_data, use_tuning=False, num_epochs=20):
    """Train model and generate credentials"""
    if use_tuning:
        # Run hyperparameter tuning
        logging.info("Starting hyperparameter tuning...")
        best_config = tune_hyperparameters(training_data)
        
        # Create pipeline with tuned parameters
        pipeline = TuneableCredentialPipeline(training_data, config=best_config)
        
        # Train with best parameters
        pipeline.train(num_epochs=num_epochs)
    else:
        # Use default parameters
        pipeline = CredentialGenerationPipeline(training_data)
        pipeline.train(num_epochs=num_epochs)

    # Generate credentials
    generated_credentials = pipeline.generate_credentials(
        num_samples=5,
        temperature=0.8,
        min_length=5
    )
    
    return generated_credentials

def main():
    parser = argparse.ArgumentParser(description='Credential Generation with Optional Hyperparameter Tuning')
    parser.add_argument('--tune', action='store_true', help='Use hyperparameter tuning')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()

    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))

    try:
        # Load credentials from CSV
        loaded_credentials = load_credentials("synthetic_credentials_snippets.csv")
        
        # Log the device being used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        torch.backends.cudnn.benchmark = True

        # Train model and generate credentials
        generated_credentials = train_and_generate(loaded_credentials, use_tuning=args.tune, num_epochs=args.epochs)
        
        # Print results with type
        print("\nGenerated Credentials:")
        for cred in generated_credentials:
            print(f"- {cred}")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()