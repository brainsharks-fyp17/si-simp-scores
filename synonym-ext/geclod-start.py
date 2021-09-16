import time

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

while True:
    try:
        request = service.instances().get(project=project, zone=zone, instance=instance)
        response = request.execute()
        # Change code below to process the `response` dict:
        status = response['status']
        if status != "RUNNING":
            request = service.instances().start(project=project, zone=zone, instance=instance)
            response = request.execute()
            print("Started")
        time.sleep(60)
    except Exception as e:
        print(e)
