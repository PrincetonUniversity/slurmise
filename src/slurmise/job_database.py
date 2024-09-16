from typing import Any, Optional
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

    def record(
        self,
        job_name: str,
        variables: dict[str, Any],
        params: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        variables: {"runtime":number of minutes,
                    "memory": number of MBs}
        """
        table = self.db.table(job_name)
        commit = table.insert(variables)
        return commit

    def query(
        self, job_name: str, params: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        table = self.db.table(job_name)
        values = table.all()
        return values

    # def delete(self, **kwargs):
    #     pass
    #
    # def clear(**kwargs):
    #     pass
    #
    # def cache_fit(ResourceFit):
    #     pass
    #
