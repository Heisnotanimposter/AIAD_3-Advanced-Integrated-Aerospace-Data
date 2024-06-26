import requests
from requests.auth import HTTPBasicAuth

# Define the URL for the dataset
url = 'https://example.com/satellite-data'

# Define the local filename to save the dataset
local_filename = 'satellite_data.zip'

# Make a GET request to download the dataset
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Write the content to a local file
    with open(local_filename, 'wb') as file:
        file.write(response.content)
    print(f'Dataset downloaded successfully as {local_filename}')
else:
    print(f'Failed to download dataset. Status code: {response.status_code}')
