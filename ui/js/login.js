var API_URL = "http://localhost:8000"
var TOKEN_ENDPOINT = "/token"


async function auth_cookie_set(options) {
    // Generate token from API and store as cookie
    const promise = await fetch(`${API_URL}${TOKEN_ENDPOINT}`, options);
    const data = await promise.json();

    if (data.detail == "Incorrect username or password") {
        return null;
    }

    // Date logic
    const minutes = 2;
    const now = new Date();
    const expiry = new Date(now.getTime() + 15 * 60000); // Convert minutes to milliseconds

    console.log(`Now: ${now}, expiry: ${expiry}`);

    // Set cookie for `minutes` mins
    // console.log(`Authorization=Bearer ${data.access_token}; expires=${expiry}`)
    console.log(expiry.toUTCString())
    document.cookie = `Authorization=Bearer ${data.access_token}; expires=${expiry.toUTCString()} ;SameSite=None; Secure; Path=/`; 
    console.log(document.cookie)

    return data

    // console.log("The token is " + data.access_token);
}


async function login(event) {
    // Get form details
    let username = document.getElementById("username").value;
    let password = document.getElementById("password").value;

    // Error check
    if ((username == "") || (password == "")) {
        alert("Please enter credentials");
        return;
    }
    
    const options = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'accept': 'application/json', // This header was in the curl command but is optional depending on your server requirements
      },
      body: new URLSearchParams({
        'grant_type': '',
        'username': username,
        'password': password, 
        'scope': '',
        'client_id': '',
        'client_secret': '',
      }).toString()
    };

    let data = await auth_cookie_set(options);
    if (data == null) {
        alert("Invalid credentials")
    } else {
        alert("Logged in for 15 minutes!");
        window.location = `${API_URL}/home`
    }
}

let submit_button = document.getElementById("submit");
submit_button.addEventListener("click", login);
