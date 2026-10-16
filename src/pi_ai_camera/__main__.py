"""Allow ``python -m pi_ai_camera`` as an alias for the ``pi-ai-camera`` CLI."""
from .main import main

raise SystemExit(main())
