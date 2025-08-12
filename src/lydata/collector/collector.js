fetch("/schema")
  .then(response => response.json())
  .then(schema => {
    // Initialize the editor with the fetched schema
    const element = document.getElementById('editor_holder');
    const options = {
      disable_edit_json: true,
      theme: 'bootstrap5',
      iconlib: 'bootstrap',
      schema: schema
    };
    var editor = new JSONEditor(element, options);

    // Hook up the submit button to log to the console
    document.getElementById('submit').addEventListener('click', function () {
      const errors = editor.validate();
      if (errors.length) {
        console.error('Validation errors:', errors);
        // display a bootstrap5 alert below the editor. Create one if it doesn't exist
        let alert = document.querySelector('.alert');
        if (!alert) {
          alert = document.createElement('div');
          alert.className = 'alert alert-danger';
          document.getElementById('editor_holder').appendChild(alert);
        }
        alert.textContent = 'Validation errors: ' + errors.map(e => e.message).join(', ');
      } else {
        // Clear any existing alert
        const existingAlert = document.querySelector('.alert');
        if (existingAlert) {
          console.log('Clearing existing alert');
          existingAlert.remove();
        }
        // Get the value from the editor
        console.log(editor.getValue());
      }
    });
  })
  .catch(error => console.error('Error loading schema:', error));
