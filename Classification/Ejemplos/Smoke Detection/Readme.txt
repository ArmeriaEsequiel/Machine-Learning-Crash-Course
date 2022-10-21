Este dataset, es acerca de las veces que un detector de humo reconocio la existencia de fuego.

Columnas:

Air Temperature
Air Humidity
TVOC: Total Volatile Organic Compounds; measured in parts per billion (Source)
eCO2: co2 equivalent concentration; calculated from different values like TVCO
Raw H2: raw molecular hydrogen; not compensated (Bias, temperature, etc.)
Raw Ethanol: raw ethanol gas (Source)
Air Pressure
PM 1.0 and PM 2.5: particulate matter size < 1.0 µm (PM1.0). 1.0 µm < 2.5 µm (PM2.5)
Fire Alarm: ground truth is "1" if a fire is there
CNT: Sample counter
UTC: Timestamp UTC seconds¶
NC0.5/NC1.0 and NC2.5: Number concentration of particulate matter. This differs from PM because NC gives the actual number of particles in the air. The raw NC is also classified by the particle size: < 0.5 µm (NC0.5); 0.5 µm < 1.0 µm (NC1.0); 1.0 µm < 2.5 µm (NC2.5)


Se realizo una clasificacion, se utilizo Pipelines para hacer el procesamiento.

Las evaluaciones se que se realizaron fueron:
1. Matriz de confusion
2. Classification_report
3. curva de ROC 
