import logging
from apscheduler.schedulers.background import BackgroundScheduler
from indexing.index_manager import index_all_tables
from indexing.qdrant_uploader import get_collection_info

logger = logging.getLogger(__name__)

# Singleton scheduler reference
_scheduler = None

def start_scheduler():
    """
    Start the background scheduler to run database re-indexing.
    - Runs a startup check: if the index is empty, runs index_all_tables immediately.
    - Registers a nightly job at 2:00 AM to perform a full recheck/re-index.
    """
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    # Initialize BackgroundScheduler
    _scheduler = BackgroundScheduler()
    
    # 1. Startup check: index if points count is 0
    try:
        col_info = get_collection_info()
        points_count = col_info.get("points_count", 0)
        logger.info(f"Startup schema index check: Found {points_count} points in Qdrant.")
        if points_count == 0:
            logger.info("Index is empty. Triggering full database schema index now...")
            # Run in a background thread or immediately during startup
            _scheduler.add_job(index_all_tables, 'date', id='startup_indexing')
    except Exception as e:
        logger.error(f"Failed to perform schema startup index check: {e}")

    # 2. Add Nightly Full Re-index Job at 2:00 AM
    _scheduler.add_job(
        index_all_tables,
        'cron',
        hour=2,
        minute=0,
        id='nightly_reindex',
        replace_existing=True
    )
    
    _scheduler.start()
    logger.info("Background indexing scheduler started successfully (nightly re-index scheduled for 02:00 AM).")
    return _scheduler
