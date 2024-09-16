import tinydb
import contextlib


class JobDatabase():
    def __init__(self, db_file):
        self.db = tinydb.TinyDB(db_file)

    def _close(self):
        self.db.close()

    @staticmethod
    @contextlib.contextmanager
    def get_database(db_file):
        db = JobDatabase(db_file)
        try:
            yield db
        finally:
            db._close()

    def record(self, job_name, **params, **variables):
        pass

    def query(self, job_name, **params):
        pass

    # def delete(self, **kwargs):
    #     pass
    #
    # def clear(**kwargs):
    #     pass
    #
    # def cache_fit(ResourceFit):
    #     pass
    #
