import time
import requests

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

credentials = GoogleCredentials.get_application_default()

service = discovery.build('compute', 'v1', credentials=credentials)

# Project ID for this request.
project = 'linux-devops-3000'

# The name of the zone for this request.
zone = 'us-central1-a'

# Name of the instance resource to return.
instance = '5299992038775316938'

url = "https://translate.google.com/?sl=en&tl=si&text=hello&op=translate"

while True:
    try:
        res = requests.request(method="GET", url=url)
        code = res.status_code
        if code != 200:
            request = service.instances().stop(project=project, zone=zone, instance=instance)
            response = request.execute()
            print("Stopped")
        time.sleep(30)
    except Exception as e:
        print(e)
