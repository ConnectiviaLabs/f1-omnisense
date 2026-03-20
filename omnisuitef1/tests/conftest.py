"""Root conftest — prevent pytest from collecting test_ functions in source modules."""

import omnihealth.timeseries

# Mark source module functions as non-test so pytest ignores them
omnihealth.timeseries.test_stationarity.__test__ = False
