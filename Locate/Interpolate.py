import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Your existing power levels and RSSI values
power_levels = np.array([10,10.25,10.5,10.75,11,11.25,11.5,11.75,12,12.25,12.5,12.75,13,13.25,13.5,13.75,14,14.25,14.5,14.75,15,15.25,15.5,15.75,16,16.25,16.5,16.75,17,17.25,17.5,17.75,18,18.25,18.5,18.75,19,19.25,19.5,19.75,20])  
rssi_values = np.array([-48.979167,-48.791667,-48.659091,-48.383333,-48.340909,-48.068182,-48.022727,-47.953125,-47.795455,-47.525,-47.214286,-47.136364,-46.954545,-46.886364,-46.619048,-46.590909,-46.545455,-46.520833,-46.479167,-46.318182,-46.115385,-45.708333,-45.613636,-45.583333,-45.479167,-45.625,-45.946429,-45.673913,-45.5,-45.568182,-45.454545,-45.134615,-45.272727,-45.295455,-44.863636,-44.840909,-44.821429,-44.788462,-44.708333,-44.576923,-44.25])   

# Create spline interpolation
spline = UnivariateSpline(power_levels, rssi_values, s=1)

# Generate interpolated values
power_interp = np.arange(10, 32.75, 0.25)
rssi_interp = spline(power_interp)

# Create DataFrame for output
df_interp = pd.DataFrame({'Power (dBm)': power_interp, 'Distance (in)': 6, 'Avg RSSI (dBm)': rssi_interp})

# Save to CSV
df_interp.to_csv("interpolated_rssi.csv", index=False)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(power_levels, rssi_values, 'o', label="Original Data", markersize=5)
plt.plot(power_interp, rssi_interp, '-', label="Interpolated Curve")
plt.xlabel("Power (dBm)")
plt.ylabel("Avg RSSI (dBm)")
plt.legend()
plt.grid()
plt.title("Interpolated RSSI vs Power")
plt.show()
