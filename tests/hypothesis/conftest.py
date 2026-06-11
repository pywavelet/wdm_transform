from hypothesis import settings

settings.register_profile("no_deadline", deadline=None)
settings.register_profile("deep", max_examples=10000)
settings.load_profile("default")  # default max_examples=100
# settings.load_profile("no_deadline")
# settings.load_profile("deep")
