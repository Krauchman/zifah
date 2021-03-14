import datetime
import copy

from rest_framework.views import APIView
from rest_framework.response import Response

from railways_ml.ml import runML


class RunML(APIView):

    def mergeCarClasses(self, wagons):
        carClasses = dict()
        for wagon in wagons:
            wagon.pop('wagonNumber')
            wagon.pop('ticketsRemaining')
            carClasses[wagon.get('carClass', '2Д')] = wagon

        return [val for key, val in carClasses.items()]

    def post(self, request):

        dates = []
        carClasses = []
        trainNumbers = []
        stations = []
        indexNumbers = []

        response_data = copy.deepcopy(request.data)
        data = request.data
        data['wagons'] = self.mergeCarClasses(data.get('wagons', []))

        for i, station in enumerate(data.get('stations', [])):
            if i == len(data.get('stations', [])) - 1:
                break
            for wagon in data.get('wagons', []):
                dates.append(data.get('date', datetime.date.today().isoformat()))
                carClasses.append(wagon.get('carClass', '2Д'))
                trainNumbers.append(data.get('trainNumber', '031Х'))
                stations.append(station.strip())
                indexNumbers.append(i + 1)

        result = runML(
            dates,
            carClasses,
            trainNumbers,
            stations,
            indexNumbers,
        )

        counts = result['Count'].astype(int)
        ticketsSolds = result['TicketsSold'].astype(int)

        response_data['predictions'] = []

        current = 0
        for i, station in enumerate(data.get('stations', [])):
            if i == len(data.get('stations', [])) - 1:
                break
            for wagon in data.get('wagons', []):
                prediction = copy.deepcopy(wagon)
                prediction['count'] = counts[current]
                prediction['ticketsSold'] = ticketsSolds[current]
                prediction['station'] = station.strip()

                response_data['predictions'].append(prediction)

                current += 1

        return Response(data=response_data)
