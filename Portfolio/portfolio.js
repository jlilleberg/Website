
  /* ---------------------- Variables --------------------------- */
  // Initialization ends on line: 37..

  let isGalleryDisplayed = true;
  let dropDownActive = false;

  let url_type = "";
  let modal_url = "";

  let project_filters = ["dl", "blog"];

  let social_navbar_text = [
    "#linkedin-navbar-text", 
    "#github-navbar-text", 
    "#medium-navbar-text", 
    "#publication-navbar-text", 
    "#twitter-navbar-text", 
    "#kaggle-navbar-text"
  ];

  let navbar_social_links = [
    "#linkedin-navbar-social-link",
    "#github-navbar-social-link",
    "#medium-navbar-social-link",
    "#publication-navbar-social-link",
    "#twitter-navbar-social-link",
    "#kaggle-navbar-social-link"
  ];

  // Project Titles
  let url2title = {
    "https://github.com/jlilleberg/Forecasting-Platinum-Palladium-Prices": "Forcasting Platinum and Palladium Prices",
    "https://github.com/jlilleberg/Malaria-Cell-Images-Classification": "Detecting Malaria Infected Bloodcells with Neural Networks",
    "https://github.com/jlilleberg/presidential-transcripts-analysis": "NLP Analysis of Presidential Transcripts",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization": "Deep Learning Specialization",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization": "Natural Language Processing Specialization",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate": "TensorFlow Developer Certificate Specialization",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization": "TensorFlow Data and Deployment Specialization"
  };

  // Project Descriptions
  let url2description = {
    "https://github.com/jlilleberg/Forecasting-Platinum-Palladium-Prices": "Having wanted to learn time-series, I took an online class, read fpp2's online forecasting book and reviewed Facebook's Prophet. I pulled the current prices of Platinum from Quandl. Since this project was to improve my ability to forecast as well as the forecasting itself, I applied as many statistical concepts as possible to reinforce the strength of my forecasts and predictions.",
    "https://github.com/jlilleberg/Malaria-Cell-Images-Classification": "Malaria is a mosquito-borne infectious disease that affects humans and other animals. The symptoms range from tiredness, vomitting, and headacches to siezures, comas, and even death. Like any disease, being able to detect if a patient is infected is desireable. The dataset consists of 150 P. falciparum-infected and 50 healthy patients collected and photographed at Chittagong Medical College Hospital, Bangladesh.",
    "https://github.com/jlilleberg/presidential-transcripts-analysis": "The motivation for this project was to analyze presidential speeches throughout American history. In this end-to-end project, I scrapped and cleaned 992 transcripts were cleaned consisting of 3.8+ million words, or 22+ million characters. I then performed multiple analyses including sentiment analysis, text generation using deep neural networks, and a multitude of visualizations.",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/Deep%20Learning%20Specialization": "...",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/NLP%20Specialization": "...",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Developer%20Certificate": "...",
    "https://github.com/jlilleberg/DeepLearning.Ai-Certifications/tree/main/TensorFlow%20Data%20and%20Deployment%20Specialization": "..."
  };

  /* ------------------- Filter Projects ------------------------ */
  
  // Displays all dl projects
  function displayDLProjects(){
    $.each(project_filters, function(index, value){

      // Fade out gallery and disable pointer
      $("#gallery").fadeOut("slow");
      $("body").css("pointer-events", "none");
      setTimeout(function(){

        // Only show selected projects | if dl else blog
        if (value == "dl") {
          $("."+value).removeClass("hide-project");
        } else {
          $("."+value).addClass("hide-project");
        }
        
        // Fade in gallery
        $("#gallery").fadeIn("slow");

        // Enable pointer
        setTimeout(function(){
          $("body").css("pointer-events", "auto");
        }, 100);
      }, 600);
    });  
  };

  // Displays all blog projects
  function displayBlogProjects(){
    $.each(project_filters, function(index, value){

      // Fade out gallery and disable pointer
      $("#gallery").fadeOut("slow");
      $("body").css("pointer-events", "none");
      setTimeout(function(){

        // Only show selected projects
        if (value == "blog") {
          $("."+value).removeClass("hide-project");
        } else {
          $("."+value).addClass("hide-project");
        }
        
        // Fade in gallery
        $("#gallery").fadeIn("slow");

        // Enable pointer
        setTimeout(function(){
          $("body").css("pointer-events", "auto");
        }, 100);      
      }, 600);
    });  
  };

  // Display all projects
  function displayAllProjects(){

     // Fade out gallery and disable pointer
    $("#gallery").fadeOut("slow");
    $("body").css("pointer-events", "none");
    setTimeout(function(){

      // Show all projects
      $(".all").removeClass("hide-project");

      // Fade in gallery
      $("#gallery").fadeIn("slow");
      
      // Enable pointer
      setTimeout(function(){
        $("body").css("pointer-events", "auto");
      }, 100);
    }, 600);
  };

  // Filter Handler
  $(function(){
    let selectedClass = "";
    $(".filter").click(function(){

      // Get respective filter
      selectedClass = $(this).attr("data-rel");

      // Fillter projects
      if (selectedClass == "all") {
        displayAllProjects();
      } else if (selectedClass == "dl") {
        displayDLProjects();
      } else if (selectedClass == "blog") {
        displayBlogProjects();
      };
    });
  });

  // Toggle boolean for if mobile navbar is active
  $(function(){
    $("#navbar-mobile-btn").click(function(){
      dropDownActive = !dropDownActive;
      if (dropDownActive) {
        $("#gallery-container").addClass("hide-gallery");
      } else {
        $("#gallery-container").removeClass("hide-gallery");
      };
    });
  });

  // Modal is clicked, blur everything else
  $(function(){
    $(".enable-blur").click(function(){
      $(".blur-candidate").addClass("blur-element");
      $(this).removeClass("blur-element");
    });
  });

  // Remove blur when modal is hidden
  $('#ModalCenter').on('hide.bs.modal', function (e) {
    $(".blur-candidate").removeClass("blur-element");
  });

  // If portfolio is clicked and we are already on portfolio, close navbar dropdown
  $(function(){
    $('#portfolio-link').click(function(){
      $("#navbar-mobile-btn").click();
    });
  });


  /* ------------------- Modal Projects Images and Links -------------------*/

  // Get and set URL for selected project
  $(function(){
    $(".project-item").click(function(){

      // Get url type and url link
      url_type = $(this).attr("url_type");
      modal_url = $(this).attr("url");

      
      $("#modal-button-github-id").attr("href", modal_url);    

      // Update modal title and modal text
      $("#modal-title-id").text(url2title[modal_url]);
      $("#modal-text-id").text(url2description[modal_url]);

      // Update Modal image to respective project image
      image_src = $(this).find("a div img").attr("src");
      $("#modal-image-id").attr("src", image_src);
    });
  });

  // Github Button is Pressed
  $(function(){
    $("#modal-button-github-id").click(function(){
      let win = window.open(modal_url, "_blank");
      if (win) {
        win.focus();
      }
    })
  })

  /* ------------------- Document is Ready ---------------------- */

  $(document).ready(function () {

    /* ---------------------- DOM is Ready -------------------------- */

    // Large Devices: Add text to social media buttons and appropriate css styles
    if($(this).width() > 1550) {
      $("#linkedin-navbar-text").text("Linkedin ");
      $("#github-navbar-text").text("Github ");
      $("#medium-navbar-text").text("Medium ");
      $("#publication-navbar-text").text("Publication ");
      $("#twitter-navbar-text").text("Twitter ");
      $("#kaggle-navbar-text").text("Kaggle ");

      // Update class for social buttons with text
      $.each(navbar_social_links, function(index, value) {
        $(value).addClass("navbar-social-btn-width-with-text");
      });

    // Medium or smaller device:
    } else {

      // Hide additional modal text for smaller screens
      $(".proj-image-text-container").addClass("hide-obj");
      $(".proj-image-title").css("top", "35%");

      // Remove text for social buttons
      $.each(social_navbar_text, function(index, value) {
        $(value).text("");
      });

      // Apply appropriate css styles
      $.each(navbar_social_links, function(index, value) {
        $(value).addClass("navbar-social-btn-width-with-no-text");
      });
    }

    /* ---------------------- Resizing ---------------------------- */

    // When resizing, check width and adjust accordingly
    $(function(){
      $(window).on('resize', function(){

        // For large devices
        if($(this).width() > 1550) {

          // Add labels to social media buttons
          $("#linkedin-navbar-text").text("Linkedin ");
          $("#github-navbar-text").text("Github ");
          $("#medium-navbar-text").text("Medium ");
          $("#publication-navbar-text").text("Publication ");
          $("#twitter-navbar-text").text("Twitter ");
          $("#kaggle-navbar-text").text("Kaggle ");

          // Apply css changes for social media buttons with text
          $.each(navbar_social_links, function(index, value){

            if ($(value).hasClass("navbar-social-btn-width-with-no-text")) {
              $(value).removeClass("navbar-social-btn-width-with-no-text");
            };

            $(value).addClass("navbar-social-btn-width-with-text");
          });

          // Display additional modal iamge text for larger screens
          $(".proj-image-text-container").removeClass("hide-obj");
          $(".proj-image-title").css("top", "20%");

        // For medium or smaller devices
        } else {

          // Hide additional modal image text for small screens
          $(".proj-image-text-container").addClass("hide-obj");
          $(".proj-image-title").css("top", "35%");

          // Remove text from social media buttons
          $.each(social_navbar_text, function(index, value) {
            $(value).text("");
          });

          // Apply css changes for social media buttons without text
          $.each(navbar_social_links, function(index, value){
            if ($(value).hasClass("navbar-social-btn-width-with-text")) {
              $(value).removeClass("navbar-social-btn-width-with-text");
            };

            $(value).addClass("navbar-social-btn-width-with-no-text");
          });
        };          
      });
    }); 

    //  If window is resized to a larger width with the navbar dropdown active, de-activate navbar dropdown
    $(function(){
      $(window).on('resize', function(){
        if($(this).width() > 991.98 && dropDownActive == true) {
          $("#navbar-mobile-btn").click();
        };
      });
    });


    $('.navbar-dropdown-toggle-button').on('click', function () {
      $('.navbar-dropdown-animation').toggleClass('open');
    });

    $(function () {
      $('[data-toggle="tooltip"]').tooltip();
    });
  });