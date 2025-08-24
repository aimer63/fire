Here is a discussion of the recommended approach using a web framework.

### Recommended Framework: FastAPI

For your project, **FastAPI** is the ideal choice. It's a modern, high-performance Python web framework that is perfectly suited for your needs for three key reasons:

1.  **Native Pydantic Integration:** This is the killer feature for you. Your entire configuration is already defined in Pydantic models (`firecast/config/config.py`). FastAPI uses Pydantic for data validation. You can use your `Config` model directly in an API endpoint, and FastAPI will automatically handle:

    - Parsing the incoming JSON request from the web form.
    - Validating the data against all your existing rules.
    - Returning clear, detailed error messages if the user's input is invalid.
    - This reuses your existing code perfectly and saves a massive amount of development time.

2.  **Automatic Interactive API Docs:** FastAPI automatically generates interactive API documentation (Swagger UI). This gives you a web page where you can test your API endpoints directly from the browser, which is invaluable for development and debugging.

3.  **Performance:** It's one of the fastest Python frameworks available, built on top of Starlette and Pydantic.

### Architectural Blueprint

You will create a classic **backend/frontend separation**. Your project will contain both the core simulation engine and the web server, but they will be run as separate processes.

#### 1. The Backend (FastAPI Application)

This will be a new Python script, let's call it `web_main.py`, in your project root. Its responsibilities are:

- **Define an API Endpoint:** You'll create an endpoint, for example `POST /api/simulate`.
- **Receive Configuration:** This endpoint will be defined to accept a request body that matches your `Config` Pydantic model. The user's inputs from the web form will be sent here as JSON.
- **Run the Simulation:** Upon receiving a valid configuration, the server will trigger the simulation run.
- **Return Results:** Once the simulation is complete, the server will send the results (success rate, final wealth data, chart points, etc.) back to the frontend as a JSON response.

**The Critical Challenge: Handling Long-Running Simulations**

A Monte Carlo simulation with 10,000 runs can take several seconds or more. A standard web request will time out, leading to a poor user experience and failed requests.

The professional solution is to use an **asynchronous task queue**.

- **How it Works:**
  1.  When the `POST /api/simulate` endpoint is called, it **does not** run the simulation itself.
  2.  Instead, it adds a "job" to a task queue (using a tool like **Celery**) and immediately returns a unique `task_id` to the user's browser.
  3.  A separate, dedicated **Worker Process** (which you also run) constantly monitors the queue. It picks up the job, runs the full simulation, and stores the results in a temporary data store (like **Redis** or a simple database), keyed by the `task_id`.
  4.  The web frontend, after receiving the `task_id`, will periodically poll a new endpoint (e.g., `GET /api/results/{task_id}`) to check the status.
  5.  Once the worker has finished, the `/api/results` endpoint will find the completed results and send them back to the frontend for display.

This architecture ensures the web interface remains responsive and can handle long-running computations without errors.

#### 2. The Frontend (The User Interface)

This part is completely separate from the backend. It consists of HTML, CSS, and JavaScript files.

- **The Form:** You will create a web form with input fields that correspond to the parameters in your `config.toml` (e.g., sliders for numbers, text boxes, tables for steps).
- **API Communication:** When the user clicks "Run Simulation," your JavaScript code will:
  1.  Gather all the data from the form.
  2.  Send it as a JSON payload in a `POST` request to your FastAPI backend's `/api/simulate` endpoint.
  3.  Receive the `task_id` and start polling the `/api/results/{task_id}` endpoint.
- **Displaying Results:** Once the results are received, the JavaScript will dynamically update the page to display the summary statistics and render interactive charts using a library like **Chart.js** or **Plotly.js**.

- **For a traditional web frontend (HTML/JS):** Libraries like **Chart.js** (MIT License) or the open-source core of **Plotly.js** (MIT License) are excellent choices.
- **For a Flutter frontend:** The best FOSS option is **`fl_chart`** (MIT License). It is highly customizable, supports a wide range of chart types, and provides the interactivity needed to visualize your simulation data effectively.

### How This Fits with Your Existing CLI

This new web interface does not interfere with your existing command-line tool at all.

- Your core logic in `firecast/core/` remains unchanged.
- Your CLI entry point, `firecast/main.py`, remains unchanged.
- You will simply have a new entry point, `web_main.py`, which you run with a web server like `uvicorn` to start the web application.

This approach gives you the best of both worlds: a scriptable, powerful CLI and an interactive, user-friendly web interface.
