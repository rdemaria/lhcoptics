_spark = None
_nxcals = None


def get_spark():
    global _spark
    if _spark is None:
        from nxcals import spark_session_builder

        _spark = spark_session_builder.get_or_create(app_name="lhcoptics")
    return _spark


def get_nxcals():
    global _nxcals
    if _nxcals is None:
        _nxcals = NXCals(spark=get_spark())
    return _nxcals


class NXCals:
    def __init__(self, spark=None):
        self.spark = spark or get_spark()
        self.api = spark._jvm.cern.nxcals.api
        self.ServiceClientFactory = (
            self.api.extraction.metadata.ServiceClientFactory
        )
        self.Variables = self.api.metadata.queries.Variables
        self.variable_service = (
            self.ServiceClientFactory.createVariableService()
        )
        self.DataQuery = self.api.extraction.data.builders.DataQuery.builder(
            self.spark
        )

    def find_variables(self, like, limit=1000):
        variables = self.variable_service.findAll(
            self.Variables.suchThat()
            .variableName()
            .like(like)
            .withOptions()
            .noConfigs()
            .orderBy()
            .variableName()
            .asc()
            .limit(limit)
        )
        return [vv.getVariableName() for vv in variables]

    def get(self, variables, start, end, system="CMW"):
        #import pytimber
        #ldb = pytimber.LoggingDB()
        #return ldb.get(variables, start, end)
        from nxcals.api.extraction.data.builders import DataQuery
        query = DataQuery.builder(self.spark).variables().system(system)
        if isinstance(variables, str):
            query = query.nameLike(variables)
        elif isinstance(variables, list):
            query = query.nameIn(variables)
        else:
            raise ValueError("variables must be a string or a list of strings")
        query = query.timeWindow(start, end).build()
        return query
