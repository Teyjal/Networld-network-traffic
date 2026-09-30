import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import requests

def test_api():
    try:
        with open('../data/sample_traffic.csv', 'rb') as f:
            files = {'file': ('sample_traffic.csv', f, 'text/csv')}
            r = requests.post('http://localhost:8000/api/traffic/upload', files=files)
        print('Traffic upload response code:', r.status_code)
        if r.status_code == 200:
            data = r.json()
            mm = data.get('forecast', {}).get('mitre_mapping', {})
            print('MITRE Mapping Stage:', mm.get('stage'))
            print('MITRE Mapping Technique:', mm.get('technique_id'), mm.get('technique_name'))
            print('MITRE Mapping Confidence:', mm.get('confidence_percent'), '%')
            print('MITRE Formatted Output:', mm.get('formatted_output'))
            print('Progression stages count:', len(mm.get('progression', [])))
            for p in mm.get('progression', []):
                print(f"  - [{p['stage']}]: Supported={p['supported']}, Conf={p['confidence_percent']}%, Rules={p['rules_satisfied']}")
                if p['supported']:
                    for ev in p['supporting_evidence']:
                        print(f"      • {ev}")
    except Exception as e:
        print("API test error:", e)

if __name__ == '__main__':
    test_api()
