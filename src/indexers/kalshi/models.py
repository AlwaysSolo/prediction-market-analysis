import re
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Optional


def parse_datetime(val: str) -> datetime:
    val = val.replace("Z", "+00:00")
    # Normalize microseconds to 6 digits
    match = re.match(r"(.+\.\d+)(\+.+)", val)
    if match:
        base, tz = match.groups()
        parts = base.split(".")
        if len(parts) == 2:
            micros = parts[1].ljust(6, "0")[:6]
            val = f"{parts[0]}.{micros}{tz}"
    return datetime.fromisoformat(val)


def parse_price_cents(value: Optional[object]) -> Optional[int]:
    """Parse either legacy cent ints or dollar strings into cents."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    return int((Decimal(str(value)) * 100).quantize(Decimal("1")))


def parse_count(value: Optional[object], fp_value: Optional[object]) -> int:
    """Parse either legacy integer count or fixed-point count string."""
    if value is not None and value != "":
        return int(value)
    if fp_value is not None and fp_value != "":
        return int(Decimal(str(fp_value)))
    return 0


def parse_optional_count(value: Optional[object], fp_value: Optional[object]) -> Optional[int]:
    if value is not None and value != "":
        return int(value)
    if fp_value is not None and fp_value != "":
        return int(Decimal(str(fp_value)))
    return None


@dataclass
class Trade:
    trade_id: str
    ticker: str
    count: int
    yes_price: int
    no_price: int
    taker_side: str
    created_time: datetime

    @classmethod
    def from_dict(cls, data: dict) -> "Trade":
        return cls(
            trade_id=data["trade_id"],
            ticker=data["ticker"],
            count=parse_count(data.get("count"), data.get("count_fp")),
            yes_price=parse_price_cents(data.get("yes_price", data.get("yes_price_dollars"))),
            no_price=parse_price_cents(data.get("no_price", data.get("no_price_dollars"))),
            taker_side=data["taker_side"],
            created_time=parse_datetime(data["created_time"]),
        )


@dataclass
class Market:
    ticker: str
    event_ticker: str
    market_type: str
    title: str
    yes_sub_title: str
    no_sub_title: str
    status: str
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    last_price: Optional[int]
    volume: int
    volume_24h: int
    open_interest: int
    result: str
    created_time: Optional[datetime]
    open_time: Optional[datetime]
    close_time: Optional[datetime]
    yes_bid_size: Optional[int] = None
    yes_ask_size: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Market":
        def parse_time(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            return parse_datetime(val)

        return cls(
            ticker=data["ticker"],
            event_ticker=data["event_ticker"],
            market_type=data.get("market_type", "binary"),
            title=data.get("title", ""),
            yes_sub_title=data.get("yes_sub_title", ""),
            no_sub_title=data.get("no_sub_title", ""),
            status=data["status"],
            yes_bid=parse_price_cents(data.get("yes_bid", data.get("yes_bid_dollars"))),
            yes_ask=parse_price_cents(data.get("yes_ask", data.get("yes_ask_dollars"))),
            no_bid=parse_price_cents(data.get("no_bid", data.get("no_bid_dollars"))),
            no_ask=parse_price_cents(data.get("no_ask", data.get("no_ask_dollars"))),
            last_price=parse_price_cents(data.get("last_price", data.get("last_price_dollars"))),
            volume=parse_count(data.get("volume"), data.get("volume_fp")),
            volume_24h=parse_count(data.get("volume_24h"), data.get("volume_24h_fp")),
            open_interest=parse_count(data.get("open_interest"), data.get("open_interest_fp")),
            result=data.get("result", ""),
            created_time=parse_time(data.get("created_time")),
            open_time=parse_time(data.get("open_time")),
            close_time=parse_time(data.get("close_time")),
            yes_bid_size=parse_optional_count(data.get("yes_bid_size"), data.get("yes_bid_size_fp")),
            yes_ask_size=parse_optional_count(data.get("yes_ask_size"), data.get("yes_ask_size_fp")),
        )
