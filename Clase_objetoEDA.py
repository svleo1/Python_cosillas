import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from IPython.display import display

class EDA:
    def __init__(self, file_path, file_type, sql_query=None):
        """
        Inicializa la clase EDA cargando un dataset desde un archivo CSV o una base de datos SQLite.

        :param file_path: Ruta al archivo CSV o SQLite.
        :param file_type: Tipo de archivo ('csv' o 'sqlite').
        :param sql_query: Consulta SQL si el archivo es SQLite.
        """
        self.df = None
        try:
            if file_type == 'csv':
                self.df = pd.read_csv(file_path)
                print("Dataset CSV cargado exitosamente.")
            elif file_type == 'sqlite' and sql_query:
                conn = sqlite3.connect(file_path)
                self.df = pd.read_sql_query(sql_query, conn)
                conn.close()
                print("Dataset SQLite cargado exitosamente.")
            else:
                raise ValueError("Tipo de archivo no soportado o consulta SQL faltante.")
        except FileNotFoundError:
            print(f"Archivo no encontrado: {file_path}")
        except sqlite3.Error as e:
            print(f"Error al ejecutar la consulta en SQLite: {e}")
        except Exception as e:
            print(f"Error inesperado al cargar el dataset: {e}")

    def show_first_last_rows(self, n=5):
        """
        Muestra las primeras y últimas n filas del DataFrame.

        :param n: Número de filas a mostrar (por defecto 5).
        """
        if self.df is not None:
            print(f"Primeras {n} filas del dataset:")
            display(self.df.head(n))
            print(f"Últimas {n} filas del dataset:")
            display(self.df.tail(n))
        else:
            print("El DataFrame no está disponible.")

    def show_dataset_shape(self):
        """
        Muestra el número de filas y columnas del DataFrame.
        """
        if self.df is not None:
            rows, cols = self.df.shape
            print(f"El dataset tiene {rows} filas y {cols} columnas.")
        else:
            print("El DataFrame no está disponible.")

    def show_column_types(self):
        """
        Muestra los nombres de las columnas y sus tipos de datos.
        """
        if self.df is not None:
            print("Tipos de datos de las columnas:")
            display(self.df.dtypes)
        else:
            print("El DataFrame no está disponible.")

    def identify_missing_values(self):
        """
        Identifica y muestra el porcentaje de valores nulos por columna.
        """
        if self.df is not None:
            total_missing = self.df.isnull().sum()
            percent_missing = (total_missing / self.df.shape[0]) * 100
            missing_data = pd.DataFrame({'Total Nulos': total_missing, 'Porcentaje Nulos': percent_missing})

            # Solo mostrar columnas con valores nulos
            missing_columns = missing_data[missing_data['Total Nulos'] > 0]
            if not missing_columns.empty:
                print("Valores nulos por columna:")
                display(missing_columns)
            else:
                print("No hay valores nulos en el dataset.")
        else:
            print("El DataFrame no está disponible.")

    def plot_missing_values(self):
        """
        Visualiza los valores nulos en el DataFrame mediante un mapa de calor.
        """
        if self.df is not None:
            if self.df.isnull().sum().sum() > 0:
                plt.figure(figsize=(12, 6))
                sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
                plt.title('Mapa de calor de valores nulos')
                plt.show()
            else:
                print("No hay valores nulos que visualizar.")
        else:
            print("El DataFrame no está disponible.")

    def detect_duplicates(self):
        """
        Detecta y muestra las filas duplicadas en el DataFrame.
        """
        if self.df is not None:
            duplicates = self.df[self.df.duplicated()]
            num_duplicates = duplicates.shape[0]
            print(f"Número de filas duplicadas: {num_duplicates}")
            if num_duplicates > 0:
                print("Filas duplicadas:")
                display(duplicates)
            else:
                print("No se encontraron filas duplicadas.")
        else:
            print("El DataFrame no está disponible.")

    def calculate_basic_statistics(self):
        """
        Muestra estadísticas descriptivas de las columnas numéricas y un resumen básico de las variables categóricas.
        """
        if self.df is not None:
            if not self.df.empty:
                print("\nEstadísticas descriptivas de variables numéricas:")
                numeric_summary = self.df.describe(include=[np.number])
                display(numeric_summary)

                print("\nResumen básico de variables categóricas:")
                categorical_summary = self.df.describe(include=['object', 'category'])
                display(categorical_summary)

                # Mostrar la moda de las variables categóricas
                print("\nModa de las variables categóricas:")
                mode_summary = self.df.select_dtypes(include=['object', 'category']).mode().iloc[0]
                display(mode_summary)
            else:
                print("El DataFrame está vacío.")
        else:
            print("El DataFrame no está disponible.")


    def unique_values_summary(self):
        """
        Muestra el número de valores únicos por columna.
        """
        if self.df is not None:
            print("Valores únicos por columna:")
            for column in self.df.columns:
                num_unique = self.df[column].nunique()
                print(f"{column}: {num_unique} valores únicos")
                if num_unique < 10:
                    print(f"- Valores: {self.df[column].unique()}")
        else:
            print("El DataFrame no está disponible.")

    def plot_numerical_kde(self):
        """
        Crea gráficos de densidad (KDE) para visualizar la distribución de variables numéricas.
        """
        if self.df is not None:
            numerical_columns = self.df.select_dtypes(include=[np.number]).columns
            if len(numerical_columns) > 0:
                # Crear una figura con múltiples gráficos KDE, uno por cada columna numérica
                plt.figure(figsize=(15, 10))
                for col in numerical_columns:
                    sns.kdeplot(self.df[col], fill=True, label=col)
                plt.title('Distribución KDE de variables numéricas')
                plt.legend()
                plt.show()
            else:
                print("No hay columnas numéricas para visualizar.")
        else:
            print("El DataFrame no está disponible.")


    def plot_boxplots(self):
        """
        Genera boxplots para variables numéricas para identificar outliers.
        """
        if self.df is not None:
            numerical_columns = self.df.select_dtypes(include=[np.number]).columns
            if len(numerical_columns) > 0:
                plt.figure(figsize=(15, 10))
                self.df[numerical_columns].boxplot()
                plt.title('Boxplots de variables numéricas')
                plt.xticks(rotation=45)
                plt.show()
            else:
                print("No hay columnas numéricas para generar boxplots.")
        else:
            print("El DataFrame no está disponible.")

    def plot_correlation_matrix(self):
        """
        Muestra la matriz de correlación de las variables numéricas.
        """
        if self.df is not None:
            numerical_columns = self.df.select_dtypes(include=[np.number]).columns
            if len(numerical_columns) > 1:
                corr_matrix = self.df[numerical_columns].corr()
                plt.figure(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
                plt.title('Matriz de correlación')
                plt.show()
            else:
                print("No hay suficientes columnas numéricas para calcular correlaciones.")
        else:
            print("El DataFrame no está disponible.")

    def plot_categorical_distribution(self, max_unique_values=50, max_columns=10):
        """
        Grafica la distribución de variables categóricas y muestra sus frecuencias.

        :param max_unique_values: Número máximo de valores únicos para graficar una columna categórica.
        :param max_columns: Número máximo de columnas categóricas que se graficarán.
        """
        if self.df is not None:
            # Selecciona columnas categóricas
            categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns

            # Filtrar columnas con más de max_unique_values valores únicos
            filtered_columns = [col for col in categorical_columns if self.df[col].nunique() <= max_unique_values]

            if len(filtered_columns) > 0:
                # Limitar el número de columnas categóricas a graficar
                filtered_columns = filtered_columns[:max_columns]
                for col in filtered_columns:
                    print(f"Distribución de '{col}':")
                    count = self.df[col].value_counts()
                    display(count)

                    plt.figure(figsize=(8, 4))
                    sns.countplot(data=self.df, y=col, order=count.index)
                    plt.title(f'Distribución de {col}')
                    plt.show()
            else:
                print(f"No hay columnas categóricas con menos de {max_unique_values} valores únicos.")
        else:
            print("El DataFrame no está disponible.")


    def detect_outliers_iqr(self):
        """
        Detecta outliers en variables numéricas utilizando el rango intercuartílico (IQR).
        """
        if self.df is not None:
            numerical_columns = self.df.select_dtypes(include=[np.number]).columns
            if len(numerical_columns) > 0:
                for col in numerical_columns:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                    num_outliers = outliers.shape[0]
                    print(f"Columna '{col}': {num_outliers} outliers detectados.")
                    if num_outliers > 0:
                        display(outliers[[col]])
            else:
                print("No hay columnas numéricas para detectar outliers.")
        else:
            print("El DataFrame no está disponible.")