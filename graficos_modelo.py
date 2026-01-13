import matplotlib.pyplot as plt
import seaborn as sns

#IMPORTANTE: TENTAR USAR SEABORN AO INVES DE MATPLOT PARA TRATAR OS DADOS

def gera_grafico_boxplot(df):

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 7, figsize=(12, 8))

    # Plot the boxplot for 'age' in the first subplot
    axes[0].boxplot(df["age"])
    axes[0].set_title("Idade")
    axes[0].set_ylabel("Anos")

    # Plot the boxplot for 'tot_bilirubin' in the second subplot
    axes[1].boxplot(df["tot_bilirubin"])
    axes[1].set_title("Bilirrubina Total")
    axes[1].set_ylabel("mg/dL")

    axes[2].boxplot(df["tot_proteins"])
    axes[2].set_title("Proteínas Totais")
    axes[2].set_ylabel("g/dL")

    axes[3].boxplot(df["albumin"])
    axes[3].set_title("Albumina")
    axes[3].set_ylabel("g/dL")

    axes[4].boxplot(df["ag_ratio"])
    axes[4].set_title("Relação Albumina/Globulina")
    axes[4].set_ylabel("g/dL")

    axes[5].boxplot(df["sgpt"])
    axes[5].set_title("SGPT (TGP/ALT)")
    axes[5].set_ylabel("u/L")

    axes[6].boxplot(df["sgot"])
    axes[6].set_title("SGOT (TGO/AST)")
    axes[6].set_ylabel("u/L")

    # Adjust layout to prevent overlapping titles/labels
    plt.tight_layout()

    # Show the plots
    plt.show()

def gera_matriz_correlacao(df):
    matriz_correlacao = df.corr()
    plt.figure(figsize=(6,4))

    sns.heatmap(matriz_correlacao, cmap='Blues', annot=True, fmt='.2f')
    plt.title('Matriz de correlação')
    plt.show()

def gera_grafico_pairplot(df):
    principal_features = ['sgot', 'sgpt', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 'albumin', 'ag_ratio']
    sns.pairplot(df[principal_features])