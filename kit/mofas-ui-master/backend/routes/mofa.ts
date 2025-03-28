import express, { Request, Response } from 'express';
import { spawn } from 'child_process';
import path from 'path';

const router = express.Router();

// Helper function to run Python scripts with arguments
function runPythonScript(scriptPath: string, args: string[] = []): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [scriptPath, ...args]);
    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python process exited with code ${code}: ${stderr}`));
      } else {
        resolve({ stdout, stderr });
      }
    });

    pythonProcess.on('error', (err) => {
      reject(err);
    });
  });
}

// Helper function to run mofa CLI commands
function runMofaCommand(command: string, args: string[] = []): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const cliPath = path.join(__dirname, '../target/cli.py');
    const allArgs = [cliPath, command, ...args];
    
    console.log('Running mofa command:', 'python', allArgs.join(' '));
    
    const pythonProcess = spawn('python', allArgs);
    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
      console.log('STDOUT:', data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
      console.error('STDERR:', data.toString());
    });

    pythonProcess.on('close', (code) => {
      console.log(`Process exited with code ${code}`);
      if (code !== 0) {
        reject(new Error(`Python process exited with code ${code}: ${stderr}`));
      } else {
        resolve({ stdout, stderr });
      }
    });

    pythonProcess.on('error', (err) => {
      console.error('Process error:', err);
      reject(err);
    });
  });
}

// GET /mofa - Main page for mofa commands
router.get('/', (req: Request, res: Response) => {
  res.render('mofa/index', { title: 'MOFA Commands' });
});

// GET /mofa/agent-list - Get list of all agents
router.get('/agent-list', async (req: Request, res: Response) => {
  try {
    const result = await runMofaCommand('agent_list');
    
    // Try to parse the output as an array if possible
    let agents = result.stdout.trim();
    try {
      // The output might be a printed array, try to parse it
      const parsedAgents = JSON.parse(agents.replace(/'/g, '"'));
      agents = parsedAgents;
    } catch (err) {
      // If parsing fails, just return the raw output
      console.log('Could not parse agent list as JSON, returning raw output');
    }

    res.json({ success: true, agents });
  } catch (err) {
    console.error('Error listing agents:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /mofa/run - Run an agent
router.post('/run', async (req: Request, res: Response) => {
  const { agentName } = req.body;
  
  if (!agentName) {
    return res.status(400).json({ success: false, error: 'Agent name is required' });
  }
  
  try {
    const args = ['--agent-name', agentName];
    const result = await runMofaCommand('run', args);
    res.json({ success: true, output: result.stdout });
  } catch (err) {
    console.error('Error running agent:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /mofa/evaluation - Run agent evaluation
router.post('/evaluation', async (req: Request, res: Response) => {
  try {
    const result = await runMofaCommand('evaluation');
    res.json({ success: true, output: result.stdout });
  } catch (err) {
    console.error('Error running evaluation:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /mofa/new-agent - Create a new agent
router.post('/new-agent', async (req: Request, res: Response) => {
  const { agentName, version, output, authors } = req.body;
  
  if (!agentName) {
    return res.status(400).json({ success: false, error: 'Agent name is required' });
  }
  
  try {
    const args = [agentName];
    if (version) args.push('--version', version);
    if (output) args.push('--output', output);
    if (authors) args.push('--authors', authors);
    
    const result = await runMofaCommand('new_agent', args);
    res.json({ success: true, output: result.stdout });
  } catch (err) {
    console.error('Error creating new agent:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// Additional route for running a custom agent with run_dspy_or_crewai_agent function
router.post('/run-custom-agent', async (req: Request, res: Response) => {
  const { agentConfig } = req.body;
  
  if (!agentConfig) {
    return res.status(400).json({ success: false, error: 'Agent configuration is required' });
  }
  
  try {
    // Create a temporary Python file that uses run_dspy_or_crewai_agent
    const scriptPath = path.join(__dirname, '../target/run_custom_agent_temp.py');
    const fs = require('fs');
    const script = `
import json
import sys
from run_agent import run_dspy_or_crewai_agent

agent_config = json.loads('''${JSON.stringify(agentConfig)}''')
result = run_dspy_or_crewai_agent(agent_config)
print(result)
`;
    
    fs.writeFileSync(scriptPath, script);
    
    try {
      const result = await runPythonScript(scriptPath);
      res.json({ success: true, output: result.stdout });
    } finally {
      // Clean up temporary file
      fs.unlinkSync(scriptPath);
    }
  } catch (err) {
    console.error('Error running custom agent:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /mofa/agent-evaluation-api - Run agent evaluation API
router.post('/agent-evaluation-api', async (req: Request, res: Response) => {
  const { primaryData, secondData, comparisonDataTask } = req.body;
  
  if (!primaryData || !secondData || !comparisonDataTask) {
    return res.status(400).json({ 
      success: false, 
      error: 'Primary data, secondary data, and comparison data task are required' 
    });
  }
  
  try {
    // Create a temporary Python file that uses agent_evaluation_api
    const scriptPath = path.join(__dirname, '../target/agent_evaluation_api_temp.py');
    const fs = require('fs');
    const script = `
import json
import sys
from agent_evealution import agent_evaluation_api

result = agent_evaluation_api(
    primary_data='''${primaryData}''',
    second_data='''${secondData}''',
    comparison_data_task='''${comparisonDataTask}'''
)
print(result)
`;
    
    fs.writeFileSync(scriptPath, script);
    
    try {
      const result = await runPythonScript(scriptPath);
      res.json({ success: true, output: result.stdout });
    } finally {
      // Clean up temporary file
      fs.unlinkSync(scriptPath);
    }
  } catch (err) {
    console.error('Error running agent evaluation API:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

export default router;