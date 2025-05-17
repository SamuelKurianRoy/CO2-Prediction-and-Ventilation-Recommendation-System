import serial
import time
from datetime import datetime
import pandas as pd
import numpy as np

class CO2Sensor:
    def __init__(self, port='COM3', baudrate=9600):
        """Initialize the CO₂ sensor connection.
        
        Args:
            port (str): Serial port for the sensor (e.g., 'COM3' for Windows, '/dev/ttyUSB0' for Linux)
            baudrate (int): Baud rate for serial communication
        """
        self.port = port
        self.baudrate = baudrate
        self.sensor = None
        self.buffer = pd.DataFrame(columns=['ts', 'co2'])
        self.buffer_size = 96  # 8 hours of data at 5-minute intervals
        
    def connect(self):
        """Establish connection with the sensor."""
        try:
            self.sensor = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to sensor on {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to sensor: {e}")
            return False
            
    def read_sensor(self):
        """Read a single measurement from the sensor.
        
        Returns:
            float: CO₂ concentration in ppm
        """
        if not self.sensor:
            raise ConnectionError("Sensor not connected")
            
        try:
            # Read data from sensor (implementation depends on your sensor's protocol)
            # This is an example - modify according to your sensor's specifications
            self.sensor.write(b'R\r\n')  # Send read command
            response = self.sensor.readline().decode().strip()
            
            # Parse the response (modify according to your sensor's output format)
            co2_value = float(response)
            return co2_value
            
        except Exception as e:
            print(f"Error reading sensor: {e}")
            return None
            
    def update_buffer(self, co2_value):
        """Update the data buffer with a new measurement."""
        current_time = datetime.now()
        
        # Add new measurement to buffer
        new_data = pd.DataFrame({
            'ts': [current_time],
            'co2': [co2_value]
        })
        
        self.buffer = pd.concat([self.buffer, new_data], ignore_index=True)
        
        # Keep only the most recent measurements
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer.tail(self.buffer_size)
            
    def get_current_data(self):
        """Get the current buffer of measurements."""
        return self.buffer.copy()
        
    def close(self):
        """Close the sensor connection."""
        if self.sensor:
            self.sensor.close()
            print("Sensor connection closed") 