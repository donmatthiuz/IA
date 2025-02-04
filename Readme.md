# Guía para ejecutar el código

## 1. Clonar el repositorio

Para obtener el código, clona el repositorio desde GitHub usando el siguiente comando:

```sh
 git clone -b lab2 https://github.com/cor22982/IA.git
```

Esto descargará el código en una carpeta llamada `IA` en tu máquina.

## 2. Crear un entorno virtual

Se recomienda usar Python 3.12. Antes de instalar las dependencias, crea un entorno virtual dentro del proyecto:

```sh
cd IA
python -m venv .venv
```

Luego, activa el entorno virtual:

- **En Windows (PowerShell):**
  ```sh
  .venv\Scripts\Activate
  ```
- **En macOS/Linux:**
  ```sh
  source .venv/bin/activate
  ```

## 3. Instalar dependencias

Una vez activado el entorno virtual, instala las dependencias con:

```sh
pip install -r requirements.txt
```

Esto instalará todas las librerías necesarias para ejecutar el código.

## 4. Codigo

Hay 2 codigos el de la task 2 y de la task 3. 

```sh
cd task_a_elegir
```
- En task2 ejecuta cada script.py
- En este es algoritmo con librerias
```sh
python lab2_librerias.py

```
- Tambien esta aqui el de manual
```sh
python lab2_manual.py
```

- En task3 ejecuta el codigo usando el siguiente comando
```sh
python task3.py
```

## 5. Si quieres ver una explicacion mas detallada de cada inciso ve a 

[Laboratorio 2](./Laboratorio2_Completado.ipynb)