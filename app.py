import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def preprocess_data(data, features, label):
    # Handle missing values
    data = data.dropna()

    # Encode categorical features
    for feature in features:
        if data[feature].dtype == 'object':
            if data[feature].nunique() < 10:
                le = LabelEncoder()
                data[feature] = le.fit_transform(data[feature])
            else:
                data = pd.get_dummies(data, columns=[feature], drop_first=True)

    # Standardize numerical features
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])

    X = data[features].values
    y = data[label].values
    
    return X, y

def main():
    st.title("ML Model Visualization")

    # File uploader for CSV files
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())

        # Feature and label selection
        columns = data.columns.tolist()
        features = st.multiselect("Select feature columns", columns)
        label = st.selectbox("Select label column", columns)

        if features and label:
            X, y = preprocess_data(data, features, label)

            model_type = st.selectbox("Select model", ["Linear Regression", "Decision Tree", "Random Forest"])
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_type == "Random Forest":
                model = RandomForestRegressor()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)

            if len(features) == 1:
                x_range = np.linspace(X.min(), X.max(), 100)
                y_range = model.predict(x_range.reshape(-1, 1))

                # Plotting with Matplotlib
                fig, ax = plt.subplots()
                ax.scatter(X, y, color='blue', alpha=0.5, label='Data points')
                ax.plot(x_range, y_range, color='red', label='Model prediction')
                ax.set_xlabel(features[0])
                ax.set_ylabel(label)
                ax.legend()

                st.pyplot(fig)

                st.write("Predicted values:", y_range)
            else:
                pca = PCA(n_components=2)
                X_reduced = pca.fit_transform(X)

                x_range = np.linspace(X_reduced[:, 0].min(), X_reduced[:, 0].max(), 100)
                y_range = np.linspace(X_reduced[:, 1].min(), X_reduced[:, 1].max(), 100)
                xx, yy = np.meshgrid(x_range, y_range)
                grid = np.c_[xx.ravel(), yy.ravel()]
                predictions = model.predict(pca.inverse_transform(grid))
                zz = predictions.reshape(xx.shape)

                # Plotting with Matplotlib
                fig, ax = plt.subplots()
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.5)
                contour = ax.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.3)
                fig.colorbar(scatter, ax=ax, label=label)

                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title('Model predictions and data points')

                st.pyplot(fig)

                st.write("Predicted values on the reduced feature space")

if __name__ == "__main__":
    main()
