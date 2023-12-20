class MismatchMembersWithStateAndExogenousDicts(BaseException):
    pass

class MissingArgument(BaseException):
    pass

class NotInitializedEnv(BaseException):
    pass

class NotEnoughExogenousData(BaseException):
    pass

class OutOfBounds(BaseException):
    pass

class ReachedTimeLimitEnv(BaseException):
    pass

class InconsistentSupport(BaseException):
    pass

class InfeasiblePolicy(BaseException):
    pass
