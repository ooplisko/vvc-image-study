"""Real-file quality matching: boundaries, gaps and rate-sensitive ties."""

import pytest

from vvcstudy.matching import actual_match, frontier


def point(qf, quality, bpp):
    return dict(parameter=qf, psnr_rgb_db=quality, bpp=bpp, file=f'{qf}.jpg', sha256=f'hash{qf}')


def test_midpoint_returns_real_file_with_smaller_rate():
    result = actual_match([point(1, 30, 1), point(2, 40, 4)], 'psnr_rgb_db', 35)
    assert result['nearest_bpp'] == 1
    assert result['nearest_file'] == '1.jpg'
    assert result['nearest_quality_delta'] == -5
    assert result['lower_bpp'] == 1 and result['upper_bpp'] == 4
    assert 'bpp' not in result  # no invented equal-quality rate


def test_nearest_upper_retains_positive_gap():
    result = actual_match([point(1, 30, 1), point(2, 40, 4)], 'psnr_rgb_db', 39)
    assert result['nearest_parameter'] == 2 and result['nearest_quality_delta'] == 1


@pytest.mark.parametrize('target,qf', [(30, 1), (40, 2)])
def test_exact_boundary_uses_one_file(target, qf):
    result = actual_match([point(1, 30, 1), point(2, 40, 4)], 'psnr_rgb_db', target)
    assert result['lower_parameter'] == result['upper_parameter'] == result['nearest_parameter'] == qf
    assert result['nearest_quality_delta'] == 0


@pytest.mark.parametrize('target', [29.99, 40.01])
def test_outside_range_is_not_filled(target):
    assert actual_match([point(1, 30, 1), point(2, 40, 4)], 'psnr_rgb_db', target) == {
        'status': 'outside_measured_range'}


def test_duplicate_quality_uses_cheapest_actual_file():
    kept, _ = frontier([point(1, 30, 2), point(2, 30, 1), point(3, 40, 4)], 'psnr_rgb_db')
    result = actual_match(kept, 'psnr_rgb_db', 30)
    assert result['nearest_file'] == '2.jpg'


def test_frontier_records_dominated_and_equivalent_points():
    kept, removed = frontier([point(3, 40, 4), point(2, 35, 5), point(1, 30, 1), point(4, 40, 4)], 'psnr_rgb_db')
    assert [row['parameter'] for row in kept] == [1, 3]
    assert removed == [{'parameter': 4, 'reason': 'equivalent'}, {'parameter': 2, 'reason': 'dominated'}]


def test_empty_frontier_is_not_extrapolated():
    assert actual_match([], 'psnr_rgb_db', 30) == {'status': 'outside_measured_range'}
