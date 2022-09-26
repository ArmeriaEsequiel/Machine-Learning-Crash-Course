Este dataset, es sobre la cantidad de electricidad generada por una planta nuclear.

Las columnas son:

Temperature in the range 1.81°C and 37.11°C,
Ambient Pressure in the range 992.89-1033.30 milibar,
Relative Humidity in the range 25.56% to 100.16%
Exhaust Vacuum in teh range 25.36-81.56 cm Hg
Net hourly electrical energy output (PE) 420.26-495.76 MW


Se realiza una regresion lineal simple y multiple. En la regresion simple se utilizan varios tipos de optimizers para comprar sus resultados.
Y en ambas es utilizado el score R2 y Mean_squared_error para tener una valoracion del modelo.

Para obtener las features, se utiliza la correlacion entre las columnas. Descartando las necesarias para evitar multicolinearidad.
