module.exports = {
  run: [
    // Clone the repository
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/Arnold2006/Jay_Caption_Beta_one_Batch_WebUI.git app",
        ]
      }
    },
    // Install Python packages
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install gradio",
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Install PyTorch (handled by torch.js)
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    }
  ]
}
