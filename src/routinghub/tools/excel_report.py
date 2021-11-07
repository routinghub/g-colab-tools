import numpy as np
import pandas as pd
import json
import os
import csv
import math
import dateutil.parser
from datetime import timedelta
from datetime import datetime, timezone
from collections import OrderedDict

## Helper functions

# si units multipliers
HOUR = 3600
KM = 1000

SECONDS = 1
MINUTES = 60 * SECONDS
RESOLUTION = 15 * MINUTES

IDLE_STATE = 'I'
WAITING_STATE = 'W'
IN_TRANSIT_STATE = 'T'
JOB_STATE = 'J'

FORMAT_LEFT_DEF = {'align': 'left', 'font_name': 'Arial'}
FORMAT_TIME_HEADER_DEF = {'font_size': 8, **FORMAT_LEFT_DEF}
FORMAT_FLOATS_LEFT_DEF = {'num_format': '0.00', **FORMAT_LEFT_DEF}

BLUE = '#78BDFF'
GREEN = '#78FFAA'
GRAY = '#F5F5F5'
DARKGRAY = '#D0D0D0'

STATE_FORMATS = {
    IDLE_STATE: {'bg_color': GRAY, **FORMAT_TIME_HEADER_DEF},
    IN_TRANSIT_STATE: {'bg_color': BLUE, **FORMAT_TIME_HEADER_DEF},
    JOB_STATE: {'bg_color': GREEN, **FORMAT_TIME_HEADER_DEF},
    WAITING_STATE: {'bg_color': DARKGRAY, **FORMAT_TIME_HEADER_DEF},
}

def add_state_formats(wb):
    return {key: wb.add_format(fmt) for key, fmt in STATE_FORMATS.items()};

def utc_to_unix(ts):
    return dateutil.parser.parse(ts).astimezone(tz=timezone.utc).timestamp()


def inside_range_step(ts, step, tr, resolution = RESOLUTION):
    return ts >= tr[0] + resolution * (step + 0) and ts <= tr[0] + resolution * (step + 1)


def downscale_ceil(ts, resolution = RESOLUTION):
    return math.ceil(ts / resolution)


def downscale_floor(ts, resolution = RESOLUTION):
    return math.floor(ts / resolution)


def format_ts_column(i, tr, resolution = RESOLUTION):
    return datetime.fromtimestamp(tr[0] + i * resolution, tz=timezone.utc).strftime('%H:%M')


def get_total_time_km_from_route(route):
    statistics = route['statistics']
    total_duration_hours = statistics['total_duration'] / HOUR
    total_km = statistics['total_travel_distance'] / KM
    return total_duration_hours, total_km


def change_datetime_format(v):
    return dateutil.parser.parse(v)


def get_statistics_from_routes(solution):
    routes = solution['result']['routes']
    statistics_by_route = pd.DataFrame(columns=[
        'Route',
        'Vehicle',
        'Total waypoints',
        'Total duration, h',
        'Total distance, km',
        'Start time',
        'Finish time'
    ])

    for i in range(len(routes)):
        route = routes[i]
        total_duration_hours = route['statistics']['total_duration'] / HOUR
        total_distance_km = route['statistics']['total_travel_distance'] / KM
        departure_time = change_datetime_format(route['waypoints'][0]['departure_time']).time()
        arrival_time = change_datetime_format(route['waypoints'][-1]['arrival_time']).time()

        route_statistics = pd.Series([
            i,
            route['vehicle']['id'],
            len(route['waypoints']),
            total_duration_hours,
            total_distance_km,
            departure_time,
            arrival_time
        ], index=[
            'Route',
            'Vehicle',
            'Total waypoints',
            'Total duration, h',
            'Total distance, km',
            'Start time',
            'Finish time'
        ])
        statistics_by_route = statistics_by_route.append(route_statistics, ignore_index=True)

    return statistics_by_route


def get_total_statistics(solution, apikey):
    stats = get_statistics_from_routes(solution)
    cols = ['Total duration, h', 'Total distance, km']
    total_sum = stats[cols].sum(axis=0)
    total_avg = stats[cols].mean(axis=0)
    total_std = stats[cols].std(axis=0)
    map_url = 'https://routinghub.com/static/tools/route.html?task_id={}&apikey={}'.format(
        solution['id'], apikey)

    def get_total_orders(waypoints):
        i = 0
        for w in waypoints:
            if 'site' in w:
                i += 1
        return i

    df = pd.DataFrame({
        'Map': [map_url],
        'Total routes': [len(solution['result']['routes'])],
        'Total vehicles': [int(solution['result']['statistics']['employed_vehicles_count'])],
        'Total sites': [sum([get_total_orders(r['waypoints']) for r in solution['result']['routes']])],
        'Total duration, h': [total_sum[0]],
        'Total distance, km': [total_sum[1]],
        'Avg duration, h': [total_avg[0]],
        'Avg distance, km': [total_avg[1]],
        'Std.dev, duration': [total_std[0]],
        'Std.dev, distance': [total_std[1]]
    }).transpose()

    df.reset_index(level=0, inplace=True)
    df.columns = ["Metric", "Value"]

    return df


def get_route_description(solution):
    rows = []
    for route_id, route in enumerate(solution['result']['routes']):
        prev_colocated = False
        for waypoint in route['waypoints']:
            arrival_time = change_datetime_format(waypoint['arrival_time'])
            departure_time = change_datetime_format(waypoint['departure_time'])
            stop_duration = timedelta(seconds=(departure_time - arrival_time).total_seconds())

            site_or_depot = waypoint['site'] if 'site' in waypoint else waypoint['depot']

            service_duration = timedelta(seconds=(site_or_depot['duration'] if 'duration' in site_or_depot else 0))
            parking_duration = timedelta(seconds=(site_or_depot['preparing_duration'] if 'preparing_duration' in site_or_depot else 0))
            if ('is_colocated' in site_or_depot) and site_or_depot['is_colocated'] == True:
                if prev_colocated:
                    parking_duration = timedelta(seconds=0)
                if not prev_colocated:
                    prev_colocated = True
            else:
                prev_colocated = False

            waiting_duration = timedelta(seconds=(stop_duration - parking_duration - service_duration).total_seconds())

            trip_duration = timedelta(seconds=round(waypoint['travel_duration']))
            waypoint_id = None

            if 'depot' in waypoint:
                waypoint_id = waypoint['depot']['id']
                coord = waypoint['depot']['location']
                job = '-'
                time_window_from = change_datetime_format(waypoint['depot']['time_window']['start'])
                time_window_to = change_datetime_format(waypoint['depot']['time_window']['end'])
            else:
                waypoint_id = waypoint['site']['id']
                coord = waypoint['site']['location']
                job = 'delivery' if 'job' not in waypoint['site'] else waypoint['site']['job']
                time_window_from = change_datetime_format(waypoint['site']['time_window']['start'])
                time_window_to = change_datetime_format(waypoint['site']['time_window']['end'])

            rows.append([
                route_id,
                route['vehicle']['id'],
                waypoint_id,
                '{:.4f},{:.4f}'.format(coord['lat'], coord['lng']),
                time_window_from.time(),
                time_window_to.time(),
                arrival_time.time(),
                departure_time.time(),
                str(stop_duration),
                str(waiting_duration),
                str(service_duration),
                str(parking_duration),
                str(trip_duration),
                job
            ])

    table = pd.DataFrame(rows)
    table.columns = [
        'Route',
        'Vehicle',
        'Stop',
        'Lat/Lng',
        'Time window from',
        'Time window to',
        'Arrival',
        'Departure',
        'Stop time',
        'Waiting time',
        'Service time',
        'Parking time',
        'Trip time',
        'Job'
    ]

    return table


def flatten_dict(dd, separator='.', prefix=''): 
    return {
        prefix + separator + k if prefix else k : v 
         for kk, vv in dd.items() 
         for k, v in flatten_dict(vv, separator, kk).items() 
    } if isinstance(dd, dict) else {prefix: dd}


def get_raw_statistics(solutions):
    stats_data = []

    for scenario_name, solution in dict(sorted(solutions.items())).items():
        stats_data.append({
            'id': scenario_name,
            'task_id': solution['id'],
            **flatten_dict(solution['result']['statistics']),
            'routes_count': len(solution['result']['routes']),
        })

    df_stats = pd.DataFrame(stats_data).transpose()
    df_stats = df_stats.rename(columns=df_stats.loc['id']).drop('id')
    return df_stats


def get_routes_timeline_df_labels(solution):
    unixtime_range = [math.inf, -math.inf] 

    for route in solution['result']['routes']:
        for waypoint in route['waypoints']:
            arrival_time = utc_to_unix(waypoint['arrival_time'])
            departure_time = utc_to_unix(waypoint['departure_time'])
            if 'depot' in waypoint:
                time_window_from = utc_to_unix(waypoint['depot']['time_window']['start'])
                time_window_to = utc_to_unix(waypoint['depot']['time_window']['end'])
            else:
                time_window_from = utc_to_unix(waypoint['site']['time_window']['start'])
                time_window_to = utc_to_unix(waypoint['site']['time_window']['end'])    

            unixtime_range[0] = min([unixtime_range[0], time_window_from, time_window_to, arrival_time, departure_time])
            unixtime_range[1] = max([unixtime_range[1], time_window_from, time_window_to, arrival_time, departure_time])

    time_range = [
        math.floor(unixtime_range[0] / RESOLUTION) * RESOLUTION,
        math.floor(unixtime_range[1] / RESOLUTION) * RESOLUTION,
    ]

    total_steps = downscale_ceil(unixtime_range[1] - unixtime_range[0])
    downscaled_begin = downscale_ceil(unixtime_range[0])

    labels_rows = []
    rows = []

    for route_id, route in enumerate(solution['result']['routes']):
        states = [IDLE_STATE] * total_steps
        labels = [None] * total_steps

        for waypoint in route['waypoints']:
            arrival_time = utc_to_unix(waypoint['arrival_time'])
            departure_time = utc_to_unix(waypoint['departure_time'])
            transit_start_time = arrival_time - waypoint['travel_duration']
            job_start_time = arrival_time + waypoint['idle_duration']
            job_end_time = job_start_time + waypoint['duration'] 

            transit_start_time = downscale_ceil(transit_start_time) - downscaled_begin - 1
            arrival_time = downscale_ceil(arrival_time) - downscaled_begin - 1
            job_start_time = downscale_ceil(job_start_time) - downscaled_begin - 1
            job_end_time = downscale_ceil(job_end_time) - downscaled_begin - 1

            if 'depot' in waypoint:
                waypoint_id = waypoint['depot']['id']
                time_window_from = change_datetime_format(waypoint['depot']['time_window']['start'])
                time_window_to = change_datetime_format(waypoint['depot']['time_window']['end'])
            else:
                waypoint_id = waypoint['site']['id']
                time_window_from = change_datetime_format(waypoint['site']['time_window']['start'])
                time_window_to = change_datetime_format(waypoint['site']['time_window']['end'])

            for i in range(transit_start_time, arrival_time):
                states[i] = IN_TRANSIT_STATE

            for i in range(arrival_time, job_start_time):
                states[i] = WAITING_STATE

            for i in range(job_start_time, job_end_time):
                states[i] = JOB_STATE

            if waypoint['arrival_time'] == waypoint['departure_time']:
                labels[arrival_time] = '{}, {}'.format(
                    waypoint_id,
                    change_datetime_format(waypoint['departure_time']).strftime('%H:%M'),
                )
            else:
                labels[arrival_time] = '{}, {}-{}'.format(
                    waypoint_id,
                    change_datetime_format(waypoint['arrival_time']).strftime('%H:%M'),
                    change_datetime_format(waypoint['departure_time']).strftime('%H:%M'),
                )

        rows.append([
            route_id,
            route['vehicle']['id'],
            route['active_shift']['id'] if 'active_shift' in route else ''
        ] + states)

        labels_rows.append(labels)

    table = pd.DataFrame(rows)
    data_columns = [
        'Route',
        'Vehicle',
        'Shift',
    ]
    table.columns = data_columns + [format_ts_column(i, time_range) for i in range(total_steps)]

    return (table, labels_rows, len(data_columns), total_steps)


def write_legend(writer, sheet_name):
    wb = writer.book
    pd.DataFrame([]).to_excel(writer, sheet_name=sheet_name)
    ws = writer.sheets[sheet_name]

    state_formats = add_state_formats(wb)
    format_left = wb.add_format(FORMAT_LEFT_DEF)

    pd.DataFrame([]).to_excel(writer, sheet_name=sheet_name)
    ws = writer.sheets[sheet_name]

    startrow = 0
    ws.write(startrow, 0, 'Legend:', format_left); startrow += 1;
    ws.write(startrow, 0, 'Idle', state_formats[IDLE_STATE]); startrow += 1;
    ws.write(startrow, 0, 'Transit', state_formats[IN_TRANSIT_STATE]); startrow += 1;
    ws.write(startrow, 0, 'Job', state_formats[JOB_STATE]); startrow += 1;
    ws.write(startrow, 0, 'Waiting', state_formats[WAITING_STATE]); startrow += 1;


def write_routes_timeline(writer, solutions, sheet_name):
    prev_header_style = pd.io.formats.excel.ExcelFormatter.header_style 
    pd.io.formats.excel.ExcelFormatter.header_style = None
    
    wb = writer.book
    
    format_left = wb.add_format(FORMAT_LEFT_DEF)
    format_time_header = wb.add_format(FORMAT_TIME_HEADER_DEF)
    state_formats = add_state_formats(wb)

    index = 1
    for task_id, solution in solutions.items():
        solution_sheet_name = '{} {}'.format(sheet_name, index); index += 1
        pd.DataFrame([]).to_excel(writer, sheet_name=solution_sheet_name)
        ws = writer.sheets[solution_sheet_name]

        startrow = 1
        ws.write('A{}'.format(startrow), task_id, format_left); startrow += 1
        
        table, labels_rows, data_cols_count, total_steps = get_routes_timeline_df_labels(solution)
        
        table.to_excel(writer,
                       sheet_name=solution_sheet_name,
                       index=False,
                       startrow=startrow)
    
        for row_index, row in table.iterrows():
            for col_index, (_, value) in list(enumerate(row.items())):
                if col_index > data_cols_count - 1:
                    if labels_rows[row_index][col_index - data_cols_count] is not None:
                        label = labels_rows[row_index][col_index - data_cols_count]
                    else:
                        label = None
                    ws.write(startrow + row_index + 1, col_index, label, state_formats[value])
            
        ws.set_column(0, data_cols_count, 10, format_left)
        ws.set_column(data_cols_count, data_cols_count + total_steps, 3.5, format_time_header)
    
    pd.io.formats.excel.ExcelFormatter.header_style = prev_header_style


def write_metrics(writer, scenario_name, solutions, apikey):
    wb = writer.book

    # merge `get_total_statistics` df for each solution in one result df
    solutions_items = iter(solutions.items())
    task_id, solution = next(solutions_items)
    df = get_total_statistics(solution, apikey)
    df.rename(columns={'Value': task_id}, inplace=True)

    for task_id, solution in solutions_items:
        df_i = get_total_statistics(solution, apikey)
        df_i.rename(columns={'Value': task_id}, inplace=True)
        df = pd.concat([df, df_i], axis=1, sort=False).T.drop_duplicates().T

    df = df.set_index('Metric').T

    # write and format
    ws_key = 'Totals {}'.format(scenario_name)
    df.to_excel(writer, sheet_name=ws_key)
    
    format_floats_left = wb.add_format(FORMAT_FLOATS_LEFT_DEF)
    format_left = wb.add_format(FORMAT_LEFT_DEF)

    ws = writer.sheets[ws_key]
    
    ws.set_column('A:A', 25, format_left)
    ws.set_column('B:C', 15, format_left)
    ws.set_column('C:D', 15, format_left)
    ws.set_column('E:J', 15, format_floats_left)


def write_routes(writer, scenario_name, solutions):
    wb = writer.book
    
    ws_key = 'Routes {}'.format(scenario_name)
    pd.DataFrame([]).to_excel(writer, sheet_name=ws_key)
    ws = writer.sheets[ws_key]

    format_floats_left = wb.add_format(FORMAT_FLOATS_LEFT_DEF)
    format_left = wb.add_format(FORMAT_LEFT_DEF)

    startrow = 0
    for task_id, solution in solutions.items():
        ws.write('A{}'.format(startrow + 1), task_id, format_left)
        df = get_statistics_from_routes(solution)
        df.to_excel(writer, 
                    sheet_name=ws_key,
                    index=False,
                    startrow=startrow + 1)
        startrow += len(df.index) + 3
    
    ws.set_column('A:A', 8, format_left)
    ws.set_column('B:C', 15, format_floats_left)
    ws.set_column('D:F', 17, format_left)
    ws.set_column('G:H', 10, format_left)


def write_waypoints(writer, scenario_name, solutions):
    wb = writer.book
    
    ws_key = 'Waypoints {}'.format(scenario_name)
    pd.DataFrame([]).to_excel(writer, sheet_name=ws_key)
    ws = writer.sheets[ws_key]

    format_floats_left = wb.add_format(FORMAT_FLOATS_LEFT_DEF)
    format_left = wb.add_format(FORMAT_LEFT_DEF)

    startrow = 0
    for task_id, solution in solutions.items():
        ws.write('A{}'.format(startrow + 1), task_id, format_left)
        df = get_route_description(solution)
        df.to_excel(writer,
                    sheet_name=ws_key,
                    index=False,
                    startrow=startrow + 1)
        startrow += len(df.index) + 3

    ws.set_column('A:A', 8, format_left)
    ws.set_column('B:B', 15, format_left)
    ws.set_column('C:D', 15, format_floats_left)
    ws.set_column('E:E', 15, format_left)
    ws.set_column('F:L', 10, format_left)


def write_report(filename, scenarios_solutions, apikey):

    pd.io.formats.excel.ExcelFormatter.header_style = {
        'font': {'name': 'Arial', 'bold': True},
        'border': 1,
        'borders': {
            'top': 'thin',
            'right': 'thin',
            'bottom': 'thin',
            'left': 'thin'
        },
    }

    with pd.ExcelWriter(filename + '.xlsx', engine='xlsxwriter') as writer:
        wb = writer.book

        for scenario_name, unsorted_solutions in scenarios_solutions.items():
            solutions = dict(sorted(unsorted_solutions.items()))
            
            print('writing metrics...')
            write_metrics(writer, scenario_name, solutions, apikey)
            print('writing routes...')
            write_routes(writer, scenario_name, solutions)
            print('writing waypoints...')
            write_waypoints(writer, scenario_name, solutions)
            print('writing legend...')
            write_legend(writer, sheet_name='Legend')
            print('writing timeline...')
            write_routes_timeline(writer, solutions, 'Timeline {}'.format(scenario_name))

    print('done')
