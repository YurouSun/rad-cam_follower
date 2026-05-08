import serial
import time

def read_serial():
    try:
        ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        print(f"Opened {ser.name} successfully.")
        
        print("Reading for 5 seconds...")
        start_time = time.time()
        while time.time() - start_time < 5:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                print(f"Received: {data.hex().upper()}")
            time.sleep(0.1)
            
        ser.close()
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_serial()
