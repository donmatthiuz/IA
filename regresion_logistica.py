import pandas as pd


file_path = "./dataset_phishing.csv"


dataset = pd.read_csv(file_path)

print("==================ANALISIS EXPLORATORIO===============")
random_domain_unique = dataset['random_domain'].unique()
port_unique = dataset['port'].unique()
dns_record_unique = dataset['dns_record'].unique()
login_form_unique = dataset['login_form'].unique()
google_index_unique = dataset['google_index'].unique()


dataset_count = len(dataset)


mean_values = dataset[['random_domain', 'port', 'dns_record', 'login_form', 'google_index']].mean(numeric_only=True)
median_values = dataset[['random_domain', 'port', 'dns_record', 'login_form', 'google_index']].median(numeric_only=True)
mode_values = dataset[['random_domain', 'port', 'dns_record', 'login_form', 'google_index']].mode().iloc[0]


print("Valores únicos para 'random_domain':")

print(random_domain_unique)
print("\nValores únicos para 'port':")

print(port_unique)

print("\nValores únicos para 'dns_record':")
print(dns_record_unique)

print("\nValores únicos para 'login_form':")
print(login_form_unique)

print("\nValores únicos para 'google_index':")
print(google_index_unique)

print(f"\nConteo total de filas en el dataset: {dataset_count}")

print("\nMedia de las columnas numéricas:")
print(mean_values)

print("\nMediana de las columnas numéricas:")
print(median_values)

print("\nModa de las columnas:")
print(mode_values)


print("==================BALANCEO===============")