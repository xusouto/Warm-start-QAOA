import os
import json
import argparse
import matplotlib.pyplot as plt

def gather_cuts(n_init, flag):
    # Define the main folder and categories
    categories = ['CS-QAOA', 'WS-QAOA', 'GW', 'RGW']
    cuts_data = {'CS-QAOA': [], 'WS-QAOA': [], 'GW': [], 'RGW': []}
    
    # Loop through each category and its corresponding solutions folder
    for category in categories:
        folder_path = os.path.join('Recursive', category, 'solutions')
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Folder for {category} not found.")
            continue
        
        # Loop through all JSON files in the solutions folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                # Extract the relevant parts from the filename
                parts = filename.split('_')
                n_init_from_file = int(parts[2][5:])  # extract n_init from 'n_init' part
                flag_from_file = parts[3] == 'True'  # extract flag (True/False)
                
                # If n_init and flag match the provided ones, add the cut_size
                if n_init_from_file == n_init and flag_from_file == flag:
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        cuts_data[category].append(data.get("cuts_size", 0))
    
    return cuts_data

def plot_histogram(cuts_data, n_init, flag):
    # Create the plot
    plt.figure(figsize=(10, 6))
    categories = ['CS-QAOA', 'WS-QAOA', 'GW', 'RGW']
    colors = ['blue', 'orange', 'green', 'red']
    
    # Plot each category's data as a histogram
    for i, category in enumerate(categories):
        plt.hist(cuts_data[category], bins=20, alpha=0.5, label=category, color=colors[i])
    
    # Add labels and title
    plt.xlabel('Cut Size')
    plt.ylabel('Number of Counts')
    plt.title(f'Histogram of Cut Sizes for n_init={n_init} and flag={flag}')
    plt.legend()
    
    # Save the histogram image
    plt.tight_layout()
    plt.savefig(f'histogram_n_init_{n_init}_flag_{flag}.png')
    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a histogram of cut sizes from solutions.")
    parser.add_argument('n_init', type=int, help="The number of nodes (n_init) of the graph.")
    parser.add_argument('flag', type=bool, help="The type of graph (True/False).")
    args = parser.parse_args()
    
    # Gather the cuts data
    cuts_data = gather_cuts(args.n_init, args.flag)
    
    # Plot the histogram
    plot_histogram(cuts_data, args.n_init, args.flag)

if __name__ == "__main__":
    main()