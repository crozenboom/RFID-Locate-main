import pandas as pd

def compute_accuracy():
    # Prompt user for the true quadrant
    valid_quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
    while True:
        true_quadrant = input("Enter the true quadrant of the tag (Q1, Q2, Q3, or Q4): ").strip().upper()
        if true_quadrant in valid_quadrants:
            break
        print("Invalid quadrant. Please enter Q1, Q2, Q3, or Q4.")

    # Define column names for live_predictions.csv
    column_names = ['log_time', 'timestamp', 'reader', 'antenna', 'epc', 'predicted_quadrant', 'confidence']

    # Read the live_predictions.csv file, assigning column names
    try:
        df = pd.read_csv('live_predictions.csv', header=None, names=column_names)
    except FileNotFoundError:
        print("Error: live_predictions.csv not found in the current directory.")
        return
    except Exception as e:
        print(f"Error reading live_predictions.csv: {str(e)}")
        return

    # Check if the file is empty
    if df.empty:
        print("Error: live_predictions.csv is empty.")
        return

    # Log the first few rows for debugging
    print(f"First 5 rows of live_predictions.csv:\n{df.head().to_string()}\n")

    # Calculate accuracy
    total_predictions = len(df)
    correct_predictions = len(df[df['predicted_quadrant'] == true_quadrant])
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0

    # Print results
    print(f"Accuracy Analysis for live_predictions.csv:")
    print(f"True Quadrant: {true_quadrant}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Correct Predictions (Q{true_quadrant[-1]}): {correct_predictions}")
    print(f"Accuracy Rate: {accuracy:.2f}%")

if __name__ == "__main__":
    compute_accuracy()