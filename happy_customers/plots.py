import matplotlib.pyplot as plt
import seaborn as sns
from happy_customers.data.load_data import load_customer_survey

def distribution_customer_happiness(customer_survey):
    #Create figure and axis with optimized size
    fig, ax = plt.subplots(figsize=(6,4))

    # Plot distribution of Target variable
    bars = customer_survey['Target'].value_counts().plot(
        kind='bar',
        color=['steelblue','orange'],
        edgecolor='black',
        width=0.6,
        ax=ax
    )

    # Set labels and title with proper formatting
    ax.set_xlabel('Target Attribute', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Distribution of Customer Happiness (1) and Unhappiness (0)', fontsize=16)
    ax.yaxis.grid(True,linestyle='--',alpha=0.7)

    # Add "Figure 1." label at the bottom of the figure
    fig.text(0.5, -0.1, 'Figure 1.', fontsize=12, ha='center')
    plt.show()

# ------------------------------------------------------------------------------------------------------
def feature_distributions(customer_survery):
    # Create figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))  # Arrange subplots in a 2x3 grid
    axes = axes.flatten()  # Flatten to easily iterate over

    # Plot histograms for each feature
    for i, column in enumerate(customer_survey.columns[1:]):  # Skip the first column (assuming it's the target)
        ax = axes[i]  # Select subplot
        customer_survey[column].hist(ax=ax, bins=5, edgecolor='black', color='steelblue', alpha=0.8)

        # Formatting
        ax.set_title(f'{column}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)  # Add grid for better readability

    # Adjust layout and add a main title
    fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust spacing to fit the title
    # Add "Figure 1." label at the bottom of the figure
    fig.text(0.5, -0.02, 'Figure 2.', fontsize=12, ha='center')
    # Show plot
    plt.show()

# ------------------------------------------------------------------------------------------------------
def correlation_plot(customer_survey):
    correlation_matrix = customer_survey.corr()
    # Plotting correlation heatmap
    fig, ax = plt.subplots(figsize=(10,7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidth=0.5,
        square=True, # heatmap cells are square
        cbar_kws={'shrink':0.8},
        ax=ax
    )
    # Set Title
    ax.set_title('Feature Correlation Heatmap', fontsize=16)
    # Tick label formatting, rotate x-axis label for better visibility
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12, rotation=0)

    # Adjust layout to prevent clipping
    fig.tight_layout()
    # Show the plot
    plt.show()

# Call the plot
if __name__ == "__main__":
    customer_survey = load_customer_survey()
    distribution_customer_happiness(customer_survey)
    feature_distributions(customer_survey)
    correlation_plot(customer_survey)