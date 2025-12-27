import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error

# --- Data Generation ---
def load_regression_data(n_samples=100):
    rng = np.random.RandomState(42)
    X = np.sort(5 * rng.rand(n_samples, 1), axis=0)
    # Creating a sine wave with some noise
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(20)) # Add some outliers/noise
    return X, y

def draw_regression_line(clf, X):
    X_grid = np.arange(min(X), max(X), 0.01)[:, np.newaxis]
    y_grid = clf.predict(X_grid)
    return X_grid, y_grid

plt.style.use('fivethirtyeight')
st.set_page_config(layout="wide", page_title="DT Regressor Playground")

# --- Sidebar Inputs ---
st.sidebar.markdown("# DTR Hyperparameters")

# Regression specific criteria: squared_error, absolute_error, friedman_mse, poisson
criterion = st.sidebar.selectbox('Criterion', ('squared_error', 'absolute_error', 'friedman_mse', 'poisson'))
splitter = st.sidebar.selectbox('Splitter', ('best', 'random'))

max_depth = st.sidebar.number_input('Max Depth (0 for None)', value=0)
max_depth = None if max_depth == 0 else max_depth

min_samples_split = st.sidebar.slider('Min Samples Split', 2, 20, 2)
min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, 20, 1)

max_features = st.sidebar.selectbox('Max Features', (None, 'sqrt', 'log2'))

max_leaf_nodes = st.sidebar.number_input('Max Leaf Nodes (0 for None)', value=0)
max_leaf_nodes = None if max_leaf_nodes == 0 else max_leaf_nodes

min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease', value=0.0, step=0.01)

# --- Logic ---
col1, col2 = st.columns(2)

X, y = load_regression_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if st.sidebar.button('Run Regressor'):
    reg = DecisionTreeRegressor(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease
    )
    
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # 1. Plot Regression Fit
    with col1:
        st.subheader("Regression Fit")
        fig, ax = plt.subplots()
        ax.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
        
        X_grid, y_grid = draw_regression_line(reg, X)
        ax.plot(X_grid, y_grid, color="cornflowerblue", label="prediction", linewidth=2)
        
        ax.set_xlabel("Data (X)")
        ax.set_ylabel("Target (y)")
        ax.legend()
        
        st.pyplot(fig)
        
        # Metrics
        st.write(f"**R2 Score:** {round(r2_score(y_test, y_pred), 3)}")
        st.write(f"**MSE:** {round(mean_squared_error(y_test, y_pred), 3)}")

    # 2. Plot Tree Structure
    with col2:
        st.subheader("Tree Structure")
        fig_tree, ax_tree = plt.subplots(figsize=(10, 8))
        plot_tree(reg, filled=True, feature_names=["X"], ax=ax_tree)
        st.pyplot(fig_tree)
else:
    with col1:
        st.info("Adjust the parameters in the sidebar and click 'Run Regressor' to see the results.")
        fig, ax = plt.subplots()
        ax.scatter(X, y, s=20, edgecolor="black", c="darkorange")
        st.pyplot(fig)