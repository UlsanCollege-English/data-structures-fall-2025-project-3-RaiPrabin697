from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import argparse
import sys
import heapq

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

MIN_LAYOVER_MINUTES = 60

# ---------------------------------------------------------
# TIME UTILITIES
# ---------------------------------------------------------

def parse_time(t: str) -> int:
    if ":" not in t:
        raise ValueError("Invalid time format")

    try:
        h, m = t.split(":")
        h, m = int(h), int(m)
    except Exception:
        raise ValueError("Invalid time format")

    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError("Invalid time range")

    return h * 60 + m


def format_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"

# ---------------------------------------------------------
# DOMAIN MODELS
# ---------------------------------------------------------

@dataclass(frozen=True)
class Flight:
    origin: str
    dest: str
    flight_number: str
    depart: int
    arrive: int
    economy: int
    business: int
    first: int

    def price_for(self, cabin: str) -> int:
        prices = {
            "economy": self.economy,
            "business": self.business,
            "first": self.first,
        }
        if cabin.lower() not in prices:
            raise ValueError("Invalid cabin")
        return prices[cabin.lower()]


class Itinerary:
    def __init__(self, flights: List[Flight]):
        self.flights = flights

    @property
    def origin(self) -> str:
        return self.flights[0].origin if self.flights else ""

    @property
    def dest(self) -> str:
        return self.flights[-1].dest if self.flights else ""

    @property
    def depart_time(self) -> int:
        return self.flights[0].depart if self.flights else 0

    @property
    def arrive_time(self) -> int:
        return self.flights[-1].arrive if self.flights else 0

    def num_stops(self) -> int:
        return max(0, len(self.flights) - 1)

    def is_empty(self) -> bool:
        return len(self.flights) == 0

    def total_price(self, cabin: str) -> int:
        return sum(f.price_for(cabin) for f in self.flights)

# ---------------------------------------------------------
# FLIGHT LOADING
# ---------------------------------------------------------

def parse_flight_line_txt(line: str) -> Optional[Flight]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) != 8:
        raise ValueError("Invalid flight line")

    origin, dest, num, dep, arr, eco, biz, fst = parts
    d = parse_time(dep)
    a = parse_time(arr)

    if a <= d:
        raise ValueError("Arrival must be after departure")

    return Flight(origin, dest, num, d, a, int(eco), int(biz), int(fst))


def load_flights_txt(path: str) -> List[Flight]:
    flights: List[Flight] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fl = parse_flight_line_txt(line)
            if fl:
                flights.append(fl)
    return flights


def load_flights_csv(path: str) -> List[Flight]:
    flights: List[Flight] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            d = parse_time(r["depart"])
            a = parse_time(r["arrive"])
            if a <= d:
                raise ValueError("Arrival must be after departure")

            flights.append(
                Flight(
                    r["origin"],
                    r["dest"],
                    r["flight_number"],
                    d,
                    a,
                    int(r["economy"]),
                    int(r["business"]),
                    int(r["first"]),
                )
            )
    return flights


def load_flights(path: str) -> List[Flight]:
    if path.endswith(".txt"):
        return load_flights_txt(path)
    if path.endswith(".csv"):
        return load_flights_csv(path)
    raise ValueError("Unsupported file type")

# ---------------------------------------------------------
# GRAPH
# ---------------------------------------------------------

def build_graph(flights: List[Flight]) -> Dict[str, List[Flight]]:
    graph: Dict[str, List[Flight]] = {}
    for f in flights:
        graph.setdefault(f.origin, []).append(f)
    return graph

# ---------------------------------------------------------
# SEARCH ALGORITHMS
# ---------------------------------------------------------

def find_earliest_itinerary(
    graph: Dict[str, List[Flight]],
    start: str,
    dest: str,
    earliest_departure: int,
) -> Optional[Itinerary]:

    pq: List[Tuple[int, List[Flight]]] = []
    best_time: Dict[str, int] = {}

    for f in graph.get(start, []):
        if f.depart >= earliest_departure:
            heapq.heappush(pq, (f.arrive, [f]))

    while pq:
        arr_time, path = heapq.heappop(pq)
        last = path[-1]

        if last.dest == dest:
            return Itinerary(path)

        if last.dest in best_time and best_time[last.dest] <= arr_time:
            continue

        best_time[last.dest] = arr_time

        for nxt in graph.get(last.dest, []):
            if nxt.depart >= last.arrive + MIN_LAYOVER_MINUTES:
                heapq.heappush(pq, (nxt.arrive, path + [nxt]))

    return None


def find_cheapest_itinerary(
    graph: Dict[str, List[Flight]],
    start: str,
    dest: str,
    earliest_departure: int,
    cabin: str,
) -> Optional[Itinerary]:

    pq: List[Tuple[int, int, List[Flight]]] = []
    best_cost: Dict[str, int] = {}

    for f in graph.get(start, []):
        if f.depart >= earliest_departure:
            heapq.heappush(pq, (f.price_for(cabin), f.arrive, [f]))

    while pq:
        cost, arr, path = heapq.heappop(pq)
        last = path[-1]

        if last.dest == dest:
            return Itinerary(path)

        if last.dest in best_cost and best_cost[last.dest] <= cost:
            continue

        best_cost[last.dest] = cost

        for nxt in graph.get(last.dest, []):
            if nxt.depart >= last.arrive + MIN_LAYOVER_MINUTES:
                heapq.heappush(
                    pq,
                    (cost + nxt.price_for(cabin), nxt.arrive, path + [nxt]),
                )

    return None

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------

@dataclass
class ComparisonRow:
    mode: str
    cabin: Optional[str]
    itinerary: Optional[Itinerary]
    note: str = ""


def format_comparison_table(
    origin: str,
    dest: str,
    earliest_departure: int,
    rows: List[ComparisonRow],
) -> str:

    out = []
    out.append(f"Route: {origin} → {dest}, Earliest depart = {format_time(earliest_departure)}")
    out.append("-" * 72)
    out.append(f"{'Mode':20} {'Cabin':8} {'Dep':6} {'Arr':6} {'Total Price':12} {'Note'}")
    out.append("-" * 72)

    for r in rows:
        if r.itinerary is None:
            out.append(f"{r.mode:20} {r.cabin or '-':8} {'-':6} {'-':6} {'-':12} {r.note}")
        else:
            itin = r.itinerary
            price = "-" if not r.cabin else str(itin.total_price(r.cabin))
            out.append(
                f"{r.mode:20} {r.cabin or '-':8} "
                f"{format_time(itin.depart_time):6} {format_time(itin.arrive_time):6} "
                f"{price:12} {r.note}"
            )

    return "\n".join(out)

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    cmp = sub.add_parser("compare")
    cmp.add_argument("file")
    cmp.add_argument("origin")
    cmp.add_argument("dest")
    cmp.add_argument("earliest")

    return parser


def main(argv=None):
    argv = argv or sys.argv[1:]
    args = build_arg_parser().parse_args(argv)

    if args.command == "compare":
        flights = load_flights(args.file)
        graph = build_graph(flights)
        earliest = parse_time(args.earliest)

        econ = find_cheapest_itinerary(graph, args.origin, args.dest, earliest, "economy")

        rows = [
            ComparisonRow(
                "Earliest arrival",
                None,
                find_earliest_itinerary(graph, args.origin, args.dest, earliest),
            ),
            ComparisonRow(
                "Cheapest (Economy)",
                "economy",
                econ,
                "no valid itinerary" if econ is None else "",
            ),
        ]

        print(format_comparison_table(args.origin, args.dest, earliest, rows))


if __name__ == "__main__":
    main()
