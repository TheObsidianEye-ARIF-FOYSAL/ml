import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
    elif dataset == "Multiclass":
        X, y = make_blobs(n_features=2, centers=3, random_state=2)
    
    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow', edgecolors='k')
    return X, y

def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.02)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.02)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

plt.style.use('fivethirtyeight')
st.set_page_config(layout="wide") # Use wide mode to show tree and plot side-by-side
st.sidebar.markdown("# DT Hyperparameters")

# --- Sidebar Inputs ---
dataset = st.sidebar.selectbox('Dataset', ('Binary', 'Multiclass'))
criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy', 'log_loss'))
splitter = st.sidebar.selectbox('Splitter', ('best', 'random'))

# Using None for "Unlimited" depth
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

fig, ax = plt.subplots()
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

if st.sidebar.button('Run Algorithm'):
    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 1. Plot Decision Boundary
    XX, YY, input_array = draw_meshgrid(X)
    labels = clf.predict(input_array)
    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.3, cmap='rainbow')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', edgecolors='k')
    
    with col1:
        st.subheader("Decision Boundary")
        st.pyplot(fig)
        st.write(f"**Accuracy:** {round(accuracy_score(y_test, y_pred), 2)}")

    # 2. Plot Tree Structure
    with col2:
        st.subheader("Tree Structure")
        fig_tree, ax_tree = plt.subplots(figsize=(10, 8))
        plot_tree(clf, filled=True, feature_names=["Col1", "Col2"], class_names=True, ax=ax_tree)
        st.pyplot(fig_tree)