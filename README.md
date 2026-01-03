# crop-yield-prediction
Machine learning project for predicting crop yields using rainfall, temperature, and pesticide data

## Overview
This project uses machine learning to predict crop yields based on environmental and agricultural factors. The analysis processes data from multiple sources including yield records, rainfall patterns, pesticide usage, and temperature measurements to train a Decision Tree Regressor model.

## Features
- Data preprocessing and cleaning for multiple agricultural datasets
- Exploratory data analysis with visualizations
- Feature engineering with one-hot encoding for categorical variables
- Decision Tree Regressor for yield prediction
- Comprehensive visualization outputs:
  - Correlation heatmaps
  - Actual vs Predicted yield plots
  - Feature importance analysis
  - Yield distribution by crop type

## Project Structure
```
crop-yield-prediction/
│
├── data/
│   ├── yield.csv          # Crop yield data by country and year
│   ├── rainfall.csv       # Average rainfall data
│   ├── pesticides.csv     # Pesticide usage data
│   └── temp.csv          # Temperature data by country
│
├── crop_yield_analysis.py # Main analysis script
├── README.md
├── .gitignore
└── requirements.txt
```

## Data Requirements
The project expects four CSV files in a `data/` directory:

1. **yield.csv** - Contains crop yield information with columns for Year, Area (country), Item (crop type), and Value
2. **rainfall.csv** - Contains average rainfall measurements per year by area
3. **pesticides.csv** - Contains pesticide usage data in tonnes by year and area
4. **temp.csv** - Contains average temperature data with 'year' and 'country' columns

## Installation

1. Clone this repository:
```bash
git clone https://github.com/sviranchi23/crop-yield-prediction.git
cd crop-yield-prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data files in the `data/` directory

## Usage

Run the analysis script:
```bash
python crop_yield_analysis.py
```

The script will:
1. Load and preprocess all data files
2. Merge datasets on common keys (Year and Area)
3. Perform exploratory data analysis
4. Train a Decision Tree Regressor model
5. Generate visualizations and save them as PNG files

## Output Files
The script generates the following visualization files:
- `correlation_heatmap.png` - Shows correlation between numerical features
- `actual_vs_predicted.png` - Scatter plot comparing model predictions with actual yields
- `feature_importance_all.png` - Bar chart showing all feature importances
- `feature_importance_top7.png` - Top 7 most important features
- `yield_by_item_boxplot.png` - Distribution of yields by crop type

## Dependencies
- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib

## Model Details
- **Algorithm**: Decision Tree Regressor
- **Features**: Rainfall, temperature, pesticide usage, and one-hot encoded country and crop types
- **Target**: Crop yield in hectograms per hectare (hg/ha)
- **Train/Test Split**: 80/20
- **Feature Scaling**: MinMaxScaler

## Notes
- The script uses relative paths (`data/*.csv`), so ensure your data files are in the correct directory
- Large datasets may require significant processing time
- The Decision Tree model can be replaced with other algorithms by modifying the model instantiation

## Future Improvements
- Add cross-validation for better model evaluation
- Implement additional regression algorithms (Random Forest, XGBoost)
- Add hyperparameter tuning
- Include more evaluation metrics (RMSE, MAE, R²)
- Create interactive visualizations

## Author
sviranchi23

## License
This project is open source and available for educational purposes.
