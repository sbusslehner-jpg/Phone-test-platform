"""In-memory backend world state (concept section 15).

The ``World`` holds the mutable state that a phonebot's tools read and write:
customers, appointments, and arbitrary business collections. It is seeded from a
scenario's ``initial_state`` before each run and its final state is compared
against the scenario's expectations after the run (``expected.database``).

State is stored as ``collection -> {id -> record}``. Records are plain dicts so
scenarios can assert on individual fields (section 20).
"""

from __future__ import annotations

import copy
from typing import Any


class World:
    """Mutable, seedable backend state for one test case."""

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        # collection name -> {record id -> record dict}
        self._collections: dict[str, dict[str, dict[str, Any]]] = {}
        # Non-collection scalars (e.g. business config) kept verbatim.
        self._scalars: dict[str, Any] = {}
        if initial_state:
            self.seed(initial_state)

    # -- seeding ---------------------------------------------------------- #

    def seed(self, initial_state: dict[str, Any]) -> None:
        """Populate the world from a scenario ``initial_state`` block.

        Two shapes are accepted per key:
        * a list of records, each with an ``id`` -> stored as a collection;
        * a single mapping with an ``id`` -> stored as a one-record collection;
        * anything else -> stored as a scalar.
        """
        for key, value in copy.deepcopy(initial_state).items():
            if isinstance(value, list) and value and all(
                isinstance(item, dict) and "id" in item for item in value
            ):
                coll = self._collections.setdefault(key, {})
                for record in value:
                    rid = self._record_id(key, record)
                    coll[rid] = record
            elif isinstance(value, dict) and "id" in value:
                coll = self._collections.setdefault(key, {})
                coll[str(value["id"])] = value
            else:
                self._scalars[key] = value

    @staticmethod
    def _record_id(collection: str, record: Any) -> str:
        if not isinstance(record, dict) or "id" not in record:
            raise ValueError(
                f"record in collection {collection!r} needs an 'id' field: {record!r}"
            )
        return str(record["id"])

    # -- access ----------------------------------------------------------- #

    def collection(self, name: str) -> dict[str, dict[str, Any]]:
        return self._collections.setdefault(name, {})

    def get(self, collection: str, record_id: str) -> dict[str, Any] | None:
        return self._collections.get(collection, {}).get(str(record_id))

    def put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        rid = self._record_id(collection, record)
        self._collections.setdefault(collection, {})[rid] = record
        return record

    def update(
        self, collection: str, record_id: str, **fields: Any
    ) -> dict[str, Any]:
        record = self.get(collection, record_id)
        if record is None:
            raise KeyError(f"{collection}/{record_id} not found")
        record.update(fields)
        return record

    def delete(self, collection: str, record_id: str) -> bool:
        coll = self._collections.get(collection, {})
        return coll.pop(str(record_id), None) is not None

    # -- snapshot / comparison ------------------------------------------- #

    def snapshot(self) -> dict[str, Any]:
        """A deep copy of the full world state (for the test report)."""
        snap: dict[str, Any] = {
            name: copy.deepcopy(records)
            for name, records in self._collections.items()
        }
        snap.update(copy.deepcopy(self._scalars))
        return snap

    def resolve(self, dotted_key: str) -> dict[str, Any] | None:
        """Resolve an ``expected.database`` key like ``"appointments.apt_42"``.

        The key format is ``"<collection>.<record_id>"``. Record ids may
        themselves contain dots/underscores, so only the first segment is
        treated as the collection name.
        """
        collection, _, record_id = dotted_key.partition(".")
        if not record_id:
            # Bare collection reference or a scalar.
            if collection in self._scalars:
                return {collection: self._scalars[collection]}
            return None
        return self.get(collection, record_id)
