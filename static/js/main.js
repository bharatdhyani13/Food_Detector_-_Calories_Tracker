$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#loader').hide();
    $('#result').hide();
    $('#show-modal').hide();
    $('#thumb').show();
    $('#result_cam').hide();
    $('#image_result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#thumb').hide();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result_cam').hide();
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();
        $('#loader').show();
        $('#show-modal').hide();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#loader').hide();
                $("#getCodeModal").modal('show');
                $('#result_cam').hide();
                $('#result').fadeIn(600);
                $('#result').html(data);
                console.log('Success!');
                $('#image_result').attr("src","");
                d = new Date();
                $('#image_result').attr("src","/static/detection.png?"+d.getTime());
                $('#image_result').show();
                $('#show-modal').show();
                abc();
            },
        });
    });


    // Webcam
    $('#btn-webcam').click(function () {
        // Make prediction by calling api /predict
        $('#result_cam').show();
        $("#fourth_row").css("padding-bottom", "100px");
        $.ajax({
            type: 'POST',
            url: '/webcam',
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // // Get and display the result
                // $('.loader').hide();
                // $('#result').fadeIn(600);
                // $('#result').html('<b> Detections:</b>  ' + data);
                // console.log('Success!');
                // $('#image_result').attr("src","");
                // d = new Date();
                // $('#image_result').attr("src","/static/detection.png?"+d.getTime());
                // $('#image_result').show();
            },
        });
    });

    $('#show-modal').click(function () {
        $("#getCodeModal").modal('show');
    });

    let searchButton = document.querySelector("#search")
    

    //Add an event listener to the button that runs the function sendApiRequest when it is clicked
    searchButton.addEventListener("click", ()=>{
        let food = document.getElementById("search_food").value;
        console.log("search button pressed")
        sendApiRequest(food)
    })
    
    async function abc() {
        $( ".custom-image" ).each((a) => {
            console.log($(this));
            console.log($(".custom-image")[a].id);
            var id = $(".custom-image")[a].id;
            $('#'+id).click(function(){
                console.log(id);
                sendApiRequest(id);
            });
        })
    }
    

    // async function buttonClicked(food) {
    //     debugger
    //     console.log(food)
    //     sendApiRequest(food)
    // }

    //An asynchronous function to fetch data from the API.
    async function sendApiRequest(food){
        let APP_ID = "7763195e"
        let APP_KEY = "20895f060f2bd27bbeace635dc0a3fd6"
        let response = await fetch(`https://api.edamam.com/search?app_id=${APP_ID}&app_key=${APP_KEY}&q=${food}`);
        // console.log(response)
        let data = await  response.json()
        console.log(data)
        useApiData(data)
    }


    //function that does something with the data received from the API. The name of the function should be customized to whatever you are doing with the data
    function useApiData(data){
        let str = ``
        for(let key = 0; key < 8; key++){
            diet_Label = data.hits[key].recipe.dietLabels[0]
            if (diet_Label == undefined){
                diet_Label = ""
            }
            // let macros = ``
            // for(let k = 0; k < 26; k++)
            // {
            //     macros = macros + `<tr>
            //             <td>${data.hits[key].recipe.digest[k].label}</td>
            //             <td>`+parseFloat(data.hits[key].recipe.digest[k].total).toFixed(2)+`</td>
            //             <td>${data.hits[key].recipe.digest[k].unit}</td>
            //             </tr>`
            // }
            str = str+`<div class="card" style="width: 18rem;margin-right: 20px;margin-bottom: 20px">
            <img class="card-img-top" src="${data.hits[key].recipe.image}" alt="Card image cap">
            <div class="card-body">
              <h5 "class="card-header">${data.hits[key].recipe.label}</h5>
              <p class="card-text"><b>Source : </b>${data.hits[key].recipe.source}</p>
              <table class="table table-striped">
                <thead>
                    <tr><b class="float-right">Calories : `+parseFloat(data.hits[key].recipe.calories).toFixed(2)+` </b> 
                    <th scope="col">Label</th>
                    <th scope="col">Quantity</th>
                    <th scope="col">Unit</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                    <td>${data.hits[key].recipe.digest[0].label}</td>
                    <td>`+parseFloat(data.hits[key].recipe.digest[0].total).toFixed(2)+`</td>
                    <td>${data.hits[key].recipe.digest[0].unit}</td>
                    </tr>
                    <tr>
                    <td>${data.hits[key].recipe.digest[1].label}</td>
                    <td>`+parseFloat(data.hits[key].recipe.digest[1].total).toFixed(2)+`</td>
                    <td>${data.hits[key].recipe.digest[1].unit}</td>
                    </tr>
                    <tr>
                    <td>${data.hits[key].recipe.digest[2].label}</td>
                    <td>`+parseFloat(data.hits[key].recipe.digest[2].total).toFixed(2)+`</td>
                    <td>${data.hits[key].recipe.digest[2].unit}</td>
                    </tr>
                </tbody>
                </table>
              <a href="${data.hits[key].recipe.url}" class="btn btn-dark">Get Recipe</a>
              <div class="float-right"><b>${diet_Label}</b></div>
            </div>
          </div>`
        }
        document.querySelector("#third_row").innerHTML = str
    }

});
