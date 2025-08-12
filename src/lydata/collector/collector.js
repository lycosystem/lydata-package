function ensureAlertExists() {
  let alert = document.querySelector('.alert');
  if (!alert) {
    alert = document.createElement('div');
    alert.className = 'alert alert-danger';
    document.getElementById('editor_holder').appendChild(alert);
  }
  return alert;
}


function ensureAlertRemoved() {
  const existingAlert = document.querySelector('.alert');
  if (existingAlert) {
    console.log('Clearing existing alert');
    existingAlert.remove();
  }
}


function ensureDownloadButtonRemoved() {
  const existingButton = document.getElementById('download_link');
  if (existingButton) {
    console.log('Clearing existing download button');
    existingButton.remove();
  }
}


function createDownloadButton(blob) {
  ensureDownloadButtonRemoved();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.id = "download_link";
  a.href = url;
  a.textContent = "Download CSV";
  a.className = "btn btn-primary";
  a.download = "lydata_records.csv";
  document.getElementById('editor_holder').appendChild(a);
  console.log('Download button created:', a);
}


function sendEditorData(editor) {
  const data = editor.getValue();
  console.log('Sending data:', data);

  fetch('/submit', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
  .then(response => response.blob())
  .then(blob => {
    console.log('Received processed data as blob:', blob);
    createDownloadButton(blob);
  });
}


function processEditor(editor) {
  const errors = editor.validate();

  if (errors.length) {
    console.error('Validation errors:', errors);
    let alert = ensureAlertExists();
    alert.textContent = 'Validation errors: ' + errors.map(e => e.message).join(', ');
  } else {
    console.log('Data successfully validated');
    ensureAlertRemoved();
    sendEditorData(editor);
  }
}


fetch("/schema")
  .then(response => response.json())
  .then(schema => {
    // Initialize the editor with the fetched schema
    const element = document.getElementById('editor_holder');
    const options = {
      disable_edit_json: true,
      theme: 'bootstrap5',
      iconlib: 'bootstrap',
      object_layout: 'grid',
      schema: schema
    };
    var editor = new JSONEditor(element, options);

    // Hook up the submit button to log to the console
    document.getElementById('submit').addEventListener('click', function() {
      console.log('Submit button clicked');
      processEditor(editor);
    });
  })
  .catch(error => console.error('Error loading schema:', error));
