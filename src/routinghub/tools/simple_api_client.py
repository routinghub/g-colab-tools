import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import json
import time
import sys
import logging
import copy
import subprocess

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Generator

def _setup_logger():
    logger = logging.getLogger('api-client')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s]: %(message)s")
    ch.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.propagate = False
    return logger


_logger = _setup_logger()


def _retryable_session():
    s = requests.Session()
    retries = Retry(total=10, backoff_factor=2)
    s.mount('https://', HTTPAdapter(max_retries=retries))
    return s


def _add_task(
    request: Dict,
    apikey: str,
    apihost: str = 'routinghub.com',
    api: str = 'routing',
    api_version: str = 'v1-devel'
) -> Dict:
    host = 'https://{}'.format(apihost)
    endpoint = '{}/api/{}/{}'.format(host, api, api_version)
    headers = {'Authorization': 'Bearer {}'.format(apikey)}
    response = _retryable_session().post('{}/add'.format(endpoint),
                                         data=json.dumps(request),
                                         headers=headers)
    return response


def get_task_result(
    task_id: str,
    apikey: str,
    apihost: str = 'routinghub.com',
    api: str = 'routing',
    api_version: str = 'v1-devel'
) -> Dict:
    host = 'https://{}'.format(apihost)
    endpoint = '{}/api/{}/{}'.format(host, api, api_version)
    headers = {'Authorization': 'Bearer {}'.format(apikey)}
    response = _retryable_session().get('{}/result/{}'.format(endpoint, task_id),
                                        headers=headers)
    return response


def _call_one(
    name: str,
    request: Dict,
    apikey: str,
    apihost: str = 'routinghub.com',
    api: str = 'routing',
    api_version: str = 'v1-devel'
) -> Dict:
    started_at = time.perf_counter()
    response = _add_task(request, apikey, apihost, api, api_version)

    task_id = response.json()['id']
    add_task_status = response.json()['status']

    _logger.info('task={} submitted: id={} status={}'.format(name, task_id, add_task_status))
    response = get_task_result(task_id, apikey,apihost,  api, api_version)

    while response.status_code not in (200, 500):
        response = get_task_result(task_id, apikey, apihost, api, api_version)
        time.sleep(5)

    finished_at = time.perf_counter()
    response_json = response.json()

    elapsed = finished_at - started_at
    _logger.info('task={} completed: status={}, elapsed={}s'.format(name, response_json['status'], elapsed))

    return response_json


def _call_many(
    named_requests: Dict[str, Dict],
    apikey: str,
    apihost: str = 'routinghub.com',
    api: str = 'routing',
    api_version: str = 'v1-devel',
    max_workers: int = 4
) -> Generator[Tuple[str, Dict], None, None]:
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_names = {}
        for name, request in named_requests.items():
            future = executor.submit(_call_one, name, request, apikey, apihost, api, api_version)
            futures_names[future] = name

        for future in as_completed(futures_names):
            name = futures_names[future]
            try:
                result = future.result()
            except Exception as exc:
                _logger.error("task={}: exception {}".format(name, exc))
            else:
                yield (name, result)

    _logger.info("done")


def _is_completed(task_id, results):
    if task_id in results:
        result = results[task_id]
        if 'status' in result and result['status'] in ('completed', 'cancelled'):
            return True
    return False


def _notify(title, message):
    command = 'osascript -e \'display notification "{}" with title "{}"\''.format(message, title)
    subprocess.run(command, shell=True)
    command = 'osascript -e \'say "{}" speaking rate 190\''.format(message)
    subprocess.run(command, shell=True)


def call_async(
    tasks: Dict[str, Dict],
    apikey: str,
    apihost: str = 'routinghub.com',
    api: str = 'routing',
    api_version: str = 'v1-devel',
    max_workers: int = 4,
    notify = False
) -> Generator[Tuple[str, Dict], None, None]:

    yield from _call_many(tasks, apikey, apihost, api, api_version, max_workers)

    if notify:
        _notify('Jupyter', 'Tasks completed')

