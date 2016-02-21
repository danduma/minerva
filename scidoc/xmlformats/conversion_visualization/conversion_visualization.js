
// All the supporting functions for visualizing SciDoc and links
//

var position_in_list=0;


function loadData(){
	
	for (var i=0; i < file_data.length; i++){
	     $('#file_list')
	         .append($("<option></option>")
	         .attr("value",i)
	         .text(file_data[i][0])); 
	}

	loadFile(position_in_list);
}


function loadFile(file_num){
	if (file_num < file_data.length){
		
		$("#leftPanel").load(file_data[file_num][0]).scrollTop();
		$("#rightPanel").load(file_data[file_num][1]).scrollTop();
	}
}

function nextFile(){
	if (position_in_list < file_data.length-1){
		position_in_list+=1;
		loadFile(position_in_list);
	}
}

function prevFile(){
	if (position_in_list > 0 && file_data.length > 0){
		position_in_list-=1;
		loadFile(position_in_list);
	}
}

function fileSelected(){
	var fl=$('#file_list');
	loadFile(fl.val());
}