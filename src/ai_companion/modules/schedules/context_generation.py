from datetime import datetime
from typing import Dict, Optional

from ai_companion.core.schedules import (
    FRIDAY_SCHEDULE,
    MONDAY_SCHEDULE,
    SATURDAY_SCHEDULE,
    SUNDAY_SCHEDULE,
    THURSDAY_SCHEDULE,
    TUESDAY_SCHEDULE,
    WEDNESDAY_SCHEDULE,
)

class ScheduleContextGenerator:
    SCHEDULES = {
        0: MONDAY_SCHEDULE,  # Monday
        1: TUESDAY_SCHEDULE,  # Tuesday
        2: WEDNESDAY_SCHEDULE,  # Wednesday
        3: THURSDAY_SCHEDULE,  # Thursday
        4: FRIDAY_SCHEDULE,  # Friday
        5: SATURDAY_SCHEDULE,  # Saturday
        6: SUNDAY_SCHEDULE,  # Sunday
    }
    
    @staticmethod
    def parse_time_range(time_range: str) -> tuple[datetime.time, datetime.time]:
        start_str, end_str = time_range.split('-')
        start_time = datetime.strptime(start_str, "%H:%M").time()
        end_time = datetime.strptime(end_str, "%H:%M").time()
        return start_time, end_time
    
    @classmethod
    def get_current_activity(cls) -> Optional[str]:
        now = datetime.now()
        weekday = now.weekday()
        current_time = now.time()
        
        daily_schedule = cls.SCHEDULES.get(weekday, {})
        
        for time_range, activity in daily_schedule.items():
            start_time, end_time = cls.parse_time_range(time_range)

            # handle overnight time ranges
            if start_time > end_time:
                if current_time >= start_time or current_time < end_time:
                    return activity
            else:
                if start_time <= current_time < end_time:
                    return activity
            return None
        
        return None
    
    @classmethod
    def get_schedule_for_day(cls, weekday: int) -> Dict[str, str]:
        return cls.SCHEDULES.get(weekday, {})