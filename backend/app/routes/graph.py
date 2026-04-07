import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from app.models.requests import ProcessDataRequest
from app.services import graph_service
from app.dependencies import get_session_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["graph"])


@router.post("/process-data")
def process_data(req: ProcessDataRequest, session_store=Depends(get_session_store)):
    """Process CSV data into a graph. Stores full graph server-side, returns lightweight summary."""
    df = pd.DataFrame(req.data)
    return graph_service.process_graph(df, req.config, session_store)


@router.get("/graph/{session_id}")
def get_graph(
    session_id: str,
    offset: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1),
    session_store=Depends(get_session_store),
):
    """Get lightweight visualization data for a session. Supports pagination via offset/limit."""
    try:
        data = graph_service.get_viz_data(session_store, session_id)
        if limit is not None:
            total_nodes = len(data["nodes"])
            total_edges = len(data["edges"])
            data["nodes"] = data["nodes"][offset:offset + limit]
            data["edges"] = data["edges"][offset:offset + limit]
            data["total_nodes"] = total_nodes
            data["total_edges"] = total_edges
        return data
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Please reprocess your graph.")


@router.get("/graph/{session_id}/stats")
def get_graph_stats(session_id: str, session_store=Depends(get_session_store)):
    """Get graph statistics without transferring full data."""
    try:
        return graph_service.get_stats(session_store, session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Please reprocess your graph.")
