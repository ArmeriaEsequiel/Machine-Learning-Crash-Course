Este data set es sobre el consumo de galones de gasolina por milla, dependiendo de cada automovil.

las columnas son:

    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)
    
    



Se realiza una regresion lineal simple y multiple. En ambas es utilizado el score R2 y Mean_squared_error para tener una valoracion del modelo.
El objetivo de este analisis, es encontrar un modelo que prediga el consumo de gasolina por milla.

Para obtener las features, se utiliza la correlacion entre las columnas. Descartando las necesarias para evitar multicolinearidad.
