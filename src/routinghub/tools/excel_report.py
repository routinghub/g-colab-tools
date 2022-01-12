import pandas as pd
import numpy as np
import math
import dateutil.parser
from datetime import timedelta
from datetime import datetime, timezone

## Helper functions

def enum(cls):
    INIT = 'init_enum'
    class T:
        def __init__(self):
            if INIT in cls.__dict__:
                cls.__dict__[INIT](cls)
        def __getattr__(self, name):
            if name == INIT: raise KeyError(INIT)
            return getattr(cls, name)
        def __getitem__(self, name):
            if name == INIT: raise KeyError(INIT)
            return getattr(cls, name)
        def __setattr__(self, name):
            raise AttributeError("type object '{}' is a const enum".format(cls, name))
        def __contains__(self, key):
            return key in self.keys()
        def keys(self):
            return [k for k, _ in cls.__dict__.items() if not k.startswith('__')]
    return T()

# si units multipliers
@enum
class UNITS:
    HOUR = 3600
    KM = 1000
    SECONDS = 1
    MINUTES = 60
    def init_enum(cls):
        cls.RESOLUTION = 15 * cls.MINUTES


@enum
class STATES:
    IDLE_STATE = 'I'
    WAITING_STATE = 'W'
    IN_TRANSIT_STATE = 'T'
    JOB_STATE = 'J'

@enum
class COLUMN_FORMATS:
    def init_enum(cls): 
        LEFT = {'align': 'left', 'font_name': 'Arial'}
        cls.LEFT = LEFT
        cls.TIME = {'font_size': 8, **LEFT}
        cls.FLOATS_LEFT = {'num_format': '0.00', **LEFT}

@enum
class COLORS:
    BLUE = '#78BDFF'
    GREEN = '#78FFAA'
    GRAY = '#F5F5F5'
    DARKGRAY = '#D0D0D0'

@enum
class STATE_FORMATS:
    IDLE_STATE = {'bg_color': COLORS.GRAY, **COLUMN_FORMATS.TIME}
    IN_TRANSIT_STATE = {'bg_color': COLORS.BLUE, **COLUMN_FORMATS.TIME}
    JOB_STATE = {'bg_color': COLORS.GREEN, **COLUMN_FORMATS.TIME}
    WAITING_STATE = {'bg_color': COLORS.DARKGRAY, **COLUMN_FORMATS.TIME}


def add_state_formats(wb):
    ret = {
        STATES[key]: wb.add_format(STATE_FORMATS[key]) for key in STATES.keys() if key in STATE_FORMATS
    }
    print(ret)
    return ret


def iso8601_to_timestamp(ts):
    return dateutil.parser.parse(ts).astimezone(tz=timezone.utc).timestamp()


def inside_range_step(ts, step, tr, resolution = UNITS.RESOLUTION):
    return ts >= tr[0] + resolution * (step + 0) and ts <= tr[0] + resolution * (step + 1)


def downscale_ceil(ts, resolution = UNITS.RESOLUTION):
    return math.ceil(ts / resolution)


def downscale_floor(ts, resolution = UNITS.RESOLUTION):
    return math.floor(ts / resolution)


def format_ts_column(i, tr, resolution = UNITS.RESOLUTION):
    return datetime.fromtimestamp(tr[0] + i * resolution, tz=timezone.utc).strftime('%H:%M')


def get_total_time_km_from_route(route):
    statistics = route['statistics']
    total_duration_hours = statistics['total_duration'] / UNITS.HOUR
    total_km = statistics['total_travel_distance'] / UNITS.KM
    return total_duration_hours, total_km


def iso8601_to_datetime(v):
    return dateutil.parser.parse(v)


def iso8601_to_short_str(v):
    return iso8601_to_datetime(v).strftime('%m/%d %H:%M:%S')


def get_route_vehicle_and_shift_id(route):
    vehicle_id = route['vehicle']['id']
    shift_id = route['active_shift']['id'] if 'active_shift' in route else ''
    return vehicle_id, shift_id


def get_waypoint_site(waypoint):
    return waypoint['site'] if 'site' in waypoint else waypoint['depot']

def get_active_shift_id(route):
    return route['active_shift']['id'] if 'active_shift' in route else ''

class RoutesStatistics:
    @enum
    class C:
        ROUTE_ID = 'Route'
        VEHICLE_ID = 'Vehicle'
        TOTAL_SITES = 'Total sites'
        TOTAL_DURATION_H = 'Total duration, h'
        TOTAL_DISTANCE_KM = 'Total distance, km'
        START_TIME = 'Start time'
        FINISH_TIME = 'Finish time'
        SHIFT_ID = 'Shift'

    def __init__(self, solution):
        routes = solution['result']['routes']
        self.route_id = []
        self.vehicle_id = []
        self.total_sites = []
        self.total_duration_h = []
        self.total_distance_km = []
        self.start_time = []
        self.finish_time = []
        self.shift_id = []
        used_shifts = []
        for i, route in enumerate(routes):
            self.route_id.append(i + 1)
            vehicle_id, shift_id = get_route_vehicle_and_shift_id(route)
            self.vehicle_id.append(vehicle_id)
            waypoints = route['waypoints']
            self.total_sites.append(int(np.sum([('site' in wp) and 1 or 0 for wp in waypoints])))
            stats = route['statistics']
            self.total_duration_h.append(stats['total_duration'] / UNITS.HOUR)
            self.total_distance_km.append(stats['total_travel_distance'] / UNITS.KM)
            self.start_time.append(iso8601_to_short_str(waypoints[0]['arrival_time']))
            self.finish_time.append(iso8601_to_short_str(waypoints[-1]['departure_time']))
            shift_id = get_active_shift_id(route)
            self.shift_id.append(shift_id)
            used_shifts.append((vehicle_id, shift_id))
        self.total_shifts = len(set(used_shifts))
        
    def to_pd(self):
        return pd.DataFrame({
            self.C.ROUTE_ID: self.route_id,
            self.C.VEHICLE_ID: self.vehicle_id,
            self.C.TOTAL_SITES: self.total_sites,
            self.C.TOTAL_DURATION_H: self.total_duration_h,
            self.C.TOTAL_DISTANCE_KM: self.total_distance_km,
            self.C.START_TIME: self.start_time,
            self.C.FINISH_TIME: self.finish_time,
            self.C.SHIFT_ID: self.shift_id,
        })


def get_statistics_from_routes(solution):
    return RoutesStatistics(solution).to_pd()


class TotalStatistics:
    @enum
    class R:
        MAP = 'Map'
        TOTAL_ROUTES = 'Total routes'
        TOTAL_VEHICLES = 'Total used vehicles'
        TOTAL_SHIFTS = 'Total used shifts'
        TOTAL_SITES = 'Total sites'
        TOTAL_DURATION_H = 'Total duration, h'
        TOTAL_DISTANCE_KM = 'Total distance, km'
        MEAN_DURATION_H = 'Mean duration, h'
        MEAN_DISTANCE_KM = 'Mean distance, km'
        STD_DURATION_H = 'Std duration, h'
        STD_DISTANCE_KM = 'Std distance, km'

    @enum
    class C:
        METRIC = 'Metric'
        VALUE = 'Value'

    MAP_URL = 'https://routinghub.com/static/tools/route.html?task_id={}&apikey={}'

    def __init__(self, solution=None, apikey=None):
        self.solution_id = solution['id']
        self.apikey = apikey
        result = solution['result']
        st = RoutesStatistics(solution)
        self.total_vehicles = int(result['statistics']['employed_vehicles_count'])
        self.total_routes = len(st.route_id)
        self.total_shifts = len(set([get_route_vehicle_and_shift_id(route) for route in result['routes']]))
        self.total_sites = int(np.sum(st.total_sites))
        self.total_duration_h = round(np.sum(st.total_duration_h), 2)
        self.total_distance_km = round(np.sum(st.total_distance_km), 2)
        self.mean_duration_h = round(np.mean(st.total_duration_h), 2)
        self.mean_distance_km = round(np.mean(st.total_distance_km), 2)
        self.std_duration_h = round(np.std(st.total_duration_h), 2)
        self.std_distance_km = round(np.std(st.total_distance_km), 2)
        
    def to_pd(self):
        df = pd.DataFrame([{
            self.R.MAP: self.MAP_URL.format(self.solution_id, self.apikey),
            self.R.TOTAL_ROUTES: self.total_routes,
            self.R.TOTAL_SHIFTS: self.total_shifts,
            self.R.TOTAL_VEHICLES: self.total_vehicles,
            self.R.TOTAL_SITES: self.total_sites,
            self.R.TOTAL_DURATION_H: self.total_duration_h,
            self.R.TOTAL_DISTANCE_KM: self.total_distance_km,
            self.R.MEAN_DURATION_H: self.mean_duration_h,
            self.R.MEAN_DISTANCE_KM: self.mean_distance_km,
            self.R.STD_DURATION_H: self.std_duration_h,
            self.R.STD_DISTANCE_KM: self.std_distance_km
        }]).transpose()
        df.reset_index(level=0, inplace=True)
        df.columns = [self.C.METRIC, self.C.VALUE]
        return df


class RouteWaypoints:
    @enum
    class C:
        ROUTE_ID = 'Route'
        VEHICLE_ID = 'Vehicle'
        SHIFT_ID = 'Vehicle'
        STOP_TYPE = 'Stop type'
        STOP_ID = 'Stop'
        COORDINATES = 'Lat/Lng'
        TW_START = 'Time window from'
        TW_END = 'Time window to'
        TRANSIT_DURATION = 'Transit duration'
        TRANSIT_DISTANCE = 'Transit distance, km'
        ARRIVAL_TIME = 'Arrival time'
        IDLE_DURATION = 'Waiting duration'
        JOB_DURATION = 'Job duration'
        DEPARTURE_TIME = 'Departure time'
        COLOCATED = 'Colocated'

    def __init__(self, route_id=None, vehicle_id=None, shift_id=None, route=None):
        waypoints = route['waypoints']
        self.route_id = [route_id] * len(waypoints)
        self.vehicle_id = [vehicle_id] * len(waypoints)
        self.shift_id = [shift_id] * len(waypoints)
        self.stop_type = []
        self.stop_id = []
        self.coordinates = []
        self.tw_start = []
        self.tw_end = []
        self.transit_duration = []
        self.transit_distance = []
        self.arrival_time = []
        self.idle_duration = []
        self.job_duration = []
        self.departure_time = []
        self.colocated_index = []
        for waypoint in waypoints:
            stop_type = 'site' if 'site' in waypoint else 'depot'
            stop_site = get_waypoint_site(waypoint)
            if stop_type == 'site':
                stop_type += ' {}'.format(stop_site.get('job', 'delivery'))
            self.stop_type.append(stop_type)
            self.stop_id.append(stop_site['id'])
            self.coordinates.append('{:.4f},{:.4f}'.format(stop_site['location']['lat'], stop_site['location']['lng']))
            self.tw_start.append(iso8601_to_short_str(stop_site['time_window']['start']))
            self.tw_end.append(iso8601_to_short_str(stop_site['time_window']['end']))
            self.transit_duration.append(str(timedelta(seconds=round(waypoint['travel_duration']))))
            self.transit_distance.append(round(waypoint['travel_distance'] / UNITS.KM, 4))
            self.arrival_time.append(iso8601_to_short_str(waypoint['arrival_time']))
            self.idle_duration.append(str(timedelta(seconds=round(waypoint['idle_duration']))))
            self.job_duration.append(str(timedelta(seconds=round(waypoint['job_duration']))))
            self.departure_time.append(iso8601_to_short_str(waypoint['departure_time']))
            self.colocated_index.append(('colocated_index' in waypoint) and waypoint['colocated_index'] or '')
        
    def to_pd(self):
        return pd.DataFrame({
            self.C.ROUTE_ID: self.route_id,
            self.C.VEHICLE_ID: self.vehicle_id,
            self.C.SHIFT_ID: self.shift_id,
            self.C.STOP_TYPE: self.stop_type,
            self.C.STOP_ID: self.stop_id,
            self.C.COORDINATES: self.coordinates,
            self.C.TW_START: self.tw_start,
            self.C.TW_END: self.tw_end,
            self.C.TRANSIT_DURATION: self.transit_duration,
            self.C.TRANSIT_DISTANCE: self.transit_distance,
            self.C.ARRIVAL_TIME: self.arrival_time,
            self.C.IDLE_DURATION: self.idle_duration,
            self.C.JOB_DURATION: self.job_duration,
            self.C.DEPARTURE_TIME: self.departure_time,
            self.C.COLOCATED: self.colocated_index
        })


class RoutesWaypoints:
    C = RouteWaypoints.C
    def __init__(self, solution=None):
        routes = solution['result']['routes']
        self.routes = []
        for i, route in enumerate(routes):
            route_id = i + 1
            vehicle_id, shift_id = get_route_vehicle_and_shift_id(route)
            self.routes.append(RouteWaypoints(route_id=route_id, vehicle_id=vehicle_id, shift_id=shift_id, route=route))
            
    def to_pd(self):
        return pd.concat([rw.to_pd() for rw in self.routes])


def get_route_description(solution):
    return RoutesWaypoints(solution).to_pd()


def get_raw_statistics(solutions):
    def flatten_dict(dd, separator='.', prefix=''): 
        return {
            prefix + separator + k if prefix else k : v 
            for kk, vv in dd.items() 
            for k, v in flatten_dict(vv, separator, kk).items() 
        } if isinstance(dd, dict) else {prefix: dd}
    stats_data = []
    for name, solution in dict(sorted(solutions.items())).items():
        stats_data.append({
            'id': name,
            'task_id': solution['id'],
            **flatten_dict(solution['result']['statistics']),
            'routes_count': len(solution['result']['routes']),
        })
    df_stats = pd.DataFrame(stats_data).transpose()
    df_stats = df_stats.rename(columns=df_stats.loc['id']).drop('id')
    return df_stats


def get_routes_timeline_df_labels(solution):
    abs_time_range = [math.inf, -math.inf] 
    for route in solution['result']['routes']:
        for waypoint in route['waypoints']:
            site = get_waypoint_site(waypoint)
            time_points_ts = [
                iso8601_to_timestamp(waypoint['arrival_time']),
                iso8601_to_timestamp(waypoint['departure_time']),
                iso8601_to_timestamp(site['time_window']['start']),
                iso8601_to_timestamp(site['time_window']['end'])
            ]
            abs_time_range[0] = min([abs_time_range[0], *time_points_ts])
            abs_time_range[1] = max([abs_time_range[1], *time_points_ts])

    time_range = [
        math.floor(abs_time_range[0] / UNITS.RESOLUTION) * UNITS.RESOLUTION,
        math.floor(abs_time_range[1] / UNITS.RESOLUTION) * UNITS.RESOLUTION,
    ]
    
    total_steps = downscale_ceil(abs_time_range[1] - abs_time_range[0])
    downscaled_begin = downscale_ceil(abs_time_range[0])
    labels_rows = []
    rows = []

    for route_id, route in enumerate(solution['result']['routes']):
        states = [STATES.IDLE_STATE] * total_steps
        labels = [None] * total_steps
        
        for waypoint in route['waypoints']:
            arrival_time_ts = iso8601_to_timestamp(waypoint['arrival_time'])
            transit_start_time_ts = arrival_time_ts - waypoint['travel_duration']
            job_start_time_ts = arrival_time_ts + waypoint['idle_duration']
            job_end_time_ts = job_start_time_ts + waypoint['job_duration'] 
            
            transit_start_time_ts = downscale_ceil(transit_start_time_ts) - downscaled_begin - 1
            arrival_time_ts = downscale_ceil(arrival_time_ts) - downscaled_begin - 1
            job_start_time_ts = downscale_ceil(job_start_time_ts) - downscaled_begin - 1
            job_end_time_ts = downscale_ceil(job_end_time_ts) - downscaled_begin - 1
            
            site = get_waypoint_site(waypoint)
            waypoint_id = site['id']
        
            for i in range(transit_start_time_ts, arrival_time_ts):
                states[i] = STATES.IN_TRANSIT_STATE
            for i in range(arrival_time_ts, job_start_time_ts):
                states[i] = STATES.WAITING_STATE
            for i in range(job_start_time_ts, job_end_time_ts):
                states[i] = STATES.JOB_STATE

            if waypoint['arrival_time'] == waypoint['departure_time']:
                labels[arrival_time_ts] = '{}, {}'.format(
                    waypoint_id,
                    iso8601_to_datetime(waypoint['departure_time']).strftime('%H:%M'),
                )
            else:
                labels[arrival_time_ts] = '{}, {}-{}'.format(
                    waypoint_id,
                    iso8601_to_datetime(waypoint['arrival_time']).strftime('%H:%M'),
                    iso8601_to_datetime(waypoint['departure_time']).strftime('%H:%M'),
                )

        rows.append([
            route_id,
            route['vehicle']['id'],
            get_active_shift_id(route)
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
    state_formats = add_state_formats(wb)
    format_left = wb.add_format(COLUMN_FORMATS.LEFT)
    pd.DataFrame([]).to_excel(writer, sheet_name=sheet_name)
    write_legend.ws = writer.sheets[sheet_name]
    write_legend.startrow = 0
    def write(title, format):
        write_legend.ws.write(write_legend.startrow, 0, title, format)
        write_legend.startrow += 1
    write('Legend:', format_left)
    write('Idle',    state_formats[STATES.IDLE_STATE])
    write('Transit', state_formats[STATES.IN_TRANSIT_STATE])
    write('Job',     state_formats[STATES.JOB_STATE])
    write('Waiting', state_formats[STATES.WAITING_STATE])


def write_routes_timeline(writer, solutions, sheet_name):
    prev_header_style = pd.io.formats.excel.ExcelFormatter.header_style
    try:
        pd.io.formats.excel.ExcelFormatter.header_style = None
        wb = writer.book
        format_left = wb.add_format(COLUMN_FORMATS.LEFT)
        format_time_header = wb.add_format(COLUMN_FORMATS.TIME)
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
    finally:
        pd.io.formats.excel.ExcelFormatter.header_style = prev_header_style


def write_metrics(writer, scenario_name, solutions, apikey):
    wb = writer.book

    # merge `get_total_statistics` df for each solution in one result df
    solutions_items = iter(solutions.items())
    task_id, solution = next(solutions_items)
    df = TotalStatistics(solution=solution, apikey=apikey).to_pd()
    df.rename(columns={'Value': task_id}, inplace=True)

    for task_id, solution in solutions_items:
        df_i = TotalStatistics(solution=solution, apikey=apikey).to_pd()
        df_i.rename(columns={'Value': task_id}, inplace=True)
        df = pd.concat([df, df_i], axis=1, sort=False).T.drop_duplicates().T

    df = df.set_index('Metric').T

    # write and format
    ws_key = 'Totals {}'.format(scenario_name)
    df.to_excel(writer, sheet_name=ws_key)
    
    format_floats_left = wb.add_format(COLUMN_FORMATS.FLOATS_LEFT)
    format_left = wb.add_format(COLUMN_FORMATS.LEFT)

    ws = writer.sheets[ws_key]
    
    ws.set_column('A:A', 25, format_left)
    ws.set_column('B:F', 15, format_left)
    ws.set_column('G:M', 15, format_floats_left)


def write_routes(writer, scenario_name, solutions):
    wb = writer.book
    
    ws_key = 'Routes {}'.format(scenario_name)
    pd.DataFrame([]).to_excel(writer, sheet_name=ws_key)
    ws = writer.sheets[ws_key]

    format_floats_left = wb.add_format(COLUMN_FORMATS.FLOATS_LEFT)
    format_left = wb.add_format(COLUMN_FORMATS.LEFT)

    startrow = 0
    for task_id, solution in solutions.items():
        ws.write('A{}'.format(startrow + 1), task_id, format_left)
        df = RoutesStatistics(solution).to_pd()
        df.to_excel(writer, 
                    sheet_name=ws_key,
                    index=False,
                    startrow=startrow + 1)
        startrow += len(df.index) + 3
    
    ws.set_column('A:A', 8, format_left)
    ws.set_column('B:C', 15, format_left)
    ws.set_column('D:E', 15, format_floats_left)
    ws.set_column('F:H', 15, format_left)


def write_waypoints(writer, scenario_name, solutions):
    wb = writer.book
    
    ws_key = 'Waypoints {}'.format(scenario_name)
    pd.DataFrame([]).to_excel(writer, sheet_name=ws_key)
    ws = writer.sheets[ws_key]

    format_floats_left = wb.add_format(COLUMN_FORMATS.FLOATS_LEFT)
    format_left = wb.add_format(COLUMN_FORMATS.LEFT)

    startrow = 0
    for task_id, solution in solutions.items():
        ws.write('A{}'.format(startrow + 1), task_id, format_left)
        df = RoutesWaypoints(solution=solution).to_pd()
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


def get_unserved(solution):
    if 'unserved' not in solution['result']:
        return None
    rows = []
    for site_id, entry in solution['result']['unserved'].items():
        site = entry['site']
        reason = entry['reason']
        coord = site['location']
        time_window_from = iso8601_to_short_str(site['time_window']['start'])
        time_window_to = iso8601_to_short_str(site['time_window']['end'])
        job_duration = timedelta(seconds=round(site.get('job_duration', 0)))
        pre_job_duration = timedelta(seconds=round(site.get('pre_job_duration', 0)))
        job = 'delivery' if 'job' not in site else site['job']

        rows.append([
            site_id,
            reason,
            '{:.4f},{:.4f}'.format(coord['lat'], coord['lng']),
            time_window_from,
            time_window_to,
            str(job_duration),
            str(pre_job_duration),
            job
        ])

    table = pd.DataFrame(rows)
    table.columns = [
        'Stop',
        'Reason',
        'Lat/Lng',
        'Time window from',
        'Time window to',
        'Job time',
        'Pre-job time',
        'Job type'
    ]
    return table


def write_unserved(writer, scenario_name, solutions):
    wb = writer.book

    ws_key = 'Unserved {}'.format(scenario_name)
    pd.DataFrame([]).to_excel(writer, sheet_name=ws_key)
    ws = writer.sheets[ws_key]

    format_floats_left = wb.add_format(COLUMN_FORMATS.FLOATS_LEFT)
    format_left = wb.add_format(COLUMN_FORMATS.LEFT)

    startrow = 0
    for task_id, solution in solutions.items():
        ws.write('A{}'.format(startrow + 1), task_id, format_left)
        df = get_unserved(solution)
        if df is not None:
            df.to_excel(writer,
                        sheet_name=ws_key,
                        index=False,
                        startrow=startrow + 1)
        startrow += len(df.index) + 3

    ws.set_column('A:A', 8, format_left)
    ws.set_column('B:C', 15, format_floats_left)
    ws.set_column('D:F', 17, format_left)
    ws.set_column('G:H', 10, format_left)


def write_report(
    filename, scenarios_solutions, apikey='', 
    metrics=True, routes=True, waypoints=True, legend=True, timeline=True, unserved=True):

    old_style = pd.io.formats.excel.ExcelFormatter.header_style
    try:
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
        filename = filename if filename.endswith('.xlsx') else filename + '.xlsx'
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            for scenario_name, unsorted_solutions in scenarios_solutions.items():
                solutions = dict(sorted(unsorted_solutions.items()))
                if metrics:
                    print('writing metrics...')
                    write_metrics(writer, scenario_name, solutions, apikey)
                if routes:
                    print('writing routes...')
                    write_routes(writer, scenario_name, solutions)
                if waypoints:
                    print('writing waypoints...')
                    write_waypoints(writer, scenario_name, solutions)
                if legend:
                    print('writing legend...')
                    write_legend(writer, sheet_name='Legend')
                if timeline:
                    print('writing timeline...')
                    write_routes_timeline(writer, solutions, 'Timeline {}'.format(scenario_name))
                if unserved:
                    print('writing unserved...')
                    write_unserved(writer, scenario_name, solutions)
    finally:
        pd.io.formats.excel.ExcelFormatter.header_style = old_style

    print('done')
