import csv

with open('C:\\Users\\caleb\\Documents\\RFID_Location\\data.csv', mode='r') as file:
    reader = csv.reader(file)

    # Skip first two rows (metadata/comments)
    next(reader)  
    next(reader)

    # Read the actual headers
    headers = next(reader)
    print("Headers:", headers)


#import csv

# Open your CSV file and check multiple rows
#with open('C:\\Users\\caleb\\Documents\\RFID_Location\\TagLogs_2025-03-03_15-13-34.csv', mode='r') as file:
#    reader = csv.reader(file)
    
    # Read the first few rows to locate headers
#    for i in range(5):  # Adjust this number if necessary
#        row = next(reader)
#        print(f"Row {i+1}: {row}")  # Print each row to see where the headers are

