import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/camera'))
from calibrate_robot import lift_preparation_status


def test_endpoint_explains_actual_adjustment_without_changing_guard():
    result=lift_preparation_status(.421999561)
    assert result['blocked'] and result['height_mm']==422.
    assert result['target_mm']==407. and result['lower_by_mm']==15.
    assert not lift_preparation_status(.407)['blocked']
    assert not lift_preparation_status(.417)['blocked']
    assert lift_preparation_status(.417001)['blocked']
