__all__ = ["init_db", "Mongo", "timer"]


import base64
import bcrypt
from contextlib import contextmanager
import pymongo
from pymongo.errors import BulkWriteError
from tails.utils import log
import time
import traceback


@contextmanager
def timer(task_description, verbose: bool = True):
    tic = time.time()
    yield
    toc = time.time()
    if verbose:
        log(f"{task_description} took {toc-tic} s")


def generate_password_hash(password, salt_rounds=12):
    password_bin = password.encode("utf-8")
    hashed = bcrypt.hashpw(password_bin, bcrypt.gensalt(salt_rounds))
    encoded = base64.b64encode(hashed)
    return encoded.decode("utf-8")


def init_db(config, verbose=False):
    """
    Initialize db if necessary: create the sole non-admin user
    """
    client = pymongo.MongoClient(
        username=config["watchdog"]["database"]["admin_username"],
        password=config["watchdog"]["database"]["admin_password"],
        host=config["watchdog"]["database"]["host"],
        port=config["watchdog"]["database"]["port"],
    )

    # _id: db_name.user_name
    user_ids = []
    for _u in client.admin.system.users.find({}, {"_id": 1}):
        user_ids.append(_u["_id"])

    db_name = config["watchdog"]["database"]["db"]
    username = config["watchdog"]["database"]["username"]

    _mongo = client[db_name]

    if f"{db_name}.{username}" not in user_ids:
        _mongo.command(
            "createUser",
            config["watchdog"]["database"]["username"],
            pwd=config["watchdog"]["database"]["password"],
            roles=["readWrite"],
        )
        if verbose:
            log("Successfully initialized db")

    _mongo.client.close()


class Mongo(object):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: str = "27017",
        username: str = None,
        password: str = None,
        db: str = None,
        verbose=0,
        **kwargs,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        self.client = pymongo.MongoClient(host=self.host, port=self.port)
        self.db = self.client[db]
        # authenticate
        self.db.authenticate(self.username, self.password)

        self.verbose = verbose

    def insert_one(
        self, collection: str, document: dict, transaction: bool = False, **kwargs
    ):
        # note to future me: single-document operations in MongoDB are atomic
        # turn on transactions only if running a replica set
        try:
            if transaction:
                with self.client.start_session() as session:
                    with session.start_transaction():
                        self.db[collection].insert_one(document, session=session)
            else:
                self.db[collection].insert_one(document)
        except Exception as e:
            if self.verbose:
                log(f"Error inserting document into collection {collection}: {str(e)}")
                err = traceback.format_exc()
                log(err)

    def insert_many(
        self, collection: str, documents: list, transaction: bool = False, **kwargs
    ):
        ordered = kwargs.get("ordered", False)
        try:
            if transaction:
                with self.client.start_session() as session:
                    with session.start_transaction():
                        self.db[collection].insert_many(
                            documents, ordered=ordered, session=session
                        )
            else:
                self.db[collection].insert_many(documents, ordered=ordered)
        except BulkWriteError as bwe:
            if self.verbose:
                log(
                    f"Error inserting documents into collection {collection}: {str(bwe.details)}"
                )
                err = traceback.format_exc()
                log(err)
        except Exception as e:
            if self.verbose:
                log(f"Error inserting documents into collection {collection}: {str(e)}")
                err = traceback.format_exc()
                log(err)

    def update_one(
        self,
        collection: str,
        filt: dict,
        update: dict,
        transaction: bool = False,
        **kwargs,
    ):
        upsert = kwargs.get("upsert", True)

        try:
            if transaction:
                with self.client.start_session() as session:
                    with session.start_transaction():
                        self.db[collection].update_one(
                            filter=filt,
                            update=update,
                            upsert=upsert,
                            session=session,
                        )
            else:
                self.db[collection].update_one(
                    filter=filt, update=update, upsert=upsert
                )
        except Exception as e:
            if self.verbose:
                log(f"Error inserting document into collection {collection}: {str(e)}")
                err = traceback.format_exc()
                log(err)
