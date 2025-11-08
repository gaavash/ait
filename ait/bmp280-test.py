import time
import board
import busio
import adafruit_bmp280

# Initialize I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize BMP280
bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(i2c)

# Optional: set sea level pressure for altitude calculation
bmp280.sea_level_pressure = 1013.25

print("Reading BMP280 sensor...")
while True:
    temperature = bmp280.temperature
    pressure = bmp280.pressure
    altitude = bmp280.altitude
    print(f"Temp: {temperature:.2f} °C | Pressure: {pressure:.2f} hPa | Altitude: {altitude:.2f} m")
    time.sleep(2)